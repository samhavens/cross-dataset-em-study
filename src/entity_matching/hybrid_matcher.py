#!/usr/bin/env python
import asyncio
import hashlib
import json
import logging
import os
import pathlib
import pickle
import random
import re
import time

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tiktoken

from tqdm import tqdm

# Fix tokenizer fork warnings in async processing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from openai import AsyncOpenAI
from src.entity_matching.constants import MODEL_COSTS
from src.entity_matching.duplicate_aware_evaluation import (
    calculate_metrics,
    duplicate_aware_detailed_evaluate,
)
from src.entity_matching.enhanced_heuristic_engine import (
    EnhancedHeuristicEngine,
    PipelineStage,
    load_enhanced_heuristics_for_dataset,
)
from src.entity_matching.experiment_config import Config, ExperimentConfig
from src.entity_matching.heuristic_engine import load_heuristics_for_dataset
from src.prompts.hybrid_matcher_prompt import build_prompt, get_prompt_data

MAX = 1_000_000  # token limit for the matching prompt



# Token counting
def token_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        # Fallback estimation
        return len(text.split()) * 1.3


def report_cost(cfg: Config):
    """Report total cost and token usage"""
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        print(f"WARNING: Model {cfg.model} not found in MODEL_COSTS. Using gpt-4.1-nano instead.")
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS["gpt-4.1-nano"]

    input_cost = (cfg.total_input_tokens / 1_000_000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1_000_000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print(f"≈{cfg.total_input_tokens / 1000:.1f}K in, {cfg.total_output_tokens / 1000:.1f}K out → ${total_cost:.3f}")

    # Return detailed token and cost information for logging
    return {
        "input_tokens": cfg.total_input_tokens,
        "output_tokens": cfg.total_output_tokens,
        "total_tokens": cfg.total_input_tokens + cfg.total_output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "model": cfg.model
    }



async def call_openai_async(prompt: str, cfg: Config, client: AsyncOpenAI) -> str:
    """Make async call to OpenAI API with exponential backoff and retries"""

    max_retries = 4
    base_delay = 2.0
    max_timeout = 300  # 5 minutes for complex entity matching prompts

    for attempt in range(max_retries):
        try:
            # Increase timeout progressively: 120s -> 180s -> 240s -> 300s
            timeout = min(120 + (attempt * 60), max_timeout)

            # o3/o4 models use max_completion_tokens instead of max_tokens and don't support temperature=0
            if cfg.model.startswith(("o3", "o4")):
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=cfg.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max(cfg.max_tokens, 1000),  # Give o4 models more tokens
                    ),
                    timeout=timeout
                )
            else:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=cfg.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                    ),
                    timeout=timeout
                )

            # Success! Track usage and return
            usage = response.usage
            cfg.total_input_tokens += usage.prompt_tokens
            cfg.total_output_tokens += usage.completion_tokens

            content = response.choices[0].message.content
            if content is None:
                print(f"  WARNING: Empty response from OpenAI API for model {cfg.model}")
                print(f"  Response object: {response}")
                return ""

            result = content.strip()
            if cfg.model.startswith(("o3", "o4")) and not result:
                print(f"  WARNING: o4 model returned empty string. Raw content: '{content}'")

            return result

        except asyncio.TimeoutError:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"OpenAI API timeout (attempt {attempt + 1}/{max_retries}) after {timeout}s for model {cfg.model}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {delay:.1f}s with {timeout + 60}s timeout...")
                await asyncio.sleep(delay)
            continue

        except Exception as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            continue

    # All retries exhausted
    print(f"❌ OpenAI API failed after {max_retries} attempts for model {cfg.model}")
    return ""


def syntactic_similarity(s1: str, s2: str) -> float:
    """Calculate syntactic similarity using difflib.SequenceMatcher"""
    from difflib import SequenceMatcher
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def trigram_similarity(s1: str, s2: str) -> float:
    """Calculate trigram similarity between two strings"""

    def get_trigrams(s):
        s = s.lower()
        return {s[i : i + 3] for i in range(len(s) - 2)}

    t1, t2 = get_trigrams(s1), get_trigrams(s2)
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def get_semantic_model(cfg: Config):
    """Get or initialize the semantic similarity model"""

    if cfg.semantic_model is None:
        print("Loading semantic similarity model (first time only)...")
        from .embedding_provider import EmbeddingProvider

        # Use API endpoint if configured, otherwise fall back to local SentenceTransformer
        if hasattr(cfg, 'embedding_base_url') and cfg.embedding_base_url:
            cfg.semantic_model = EmbeddingProvider(cfg.embedding_model_name, base_url=cfg.embedding_base_url)
        else:
            cfg.semantic_model = EmbeddingProvider(cfg.embedding_model_name)

    return cfg.semantic_model


def get_heuristic_engine(cfg: Config, dataset: str):
    """Get or initialize the heuristic engine"""
    if not cfg.use_heuristics:
        return None

    if cfg.heuristic_engine is None:
        print("Loading heuristic engine (first time only)...")
        cfg.heuristic_engine = load_heuristics_for_dataset(dataset, cfg.heuristic_file)

        if cfg.heuristic_engine.rules:
            print(f"Loaded {len(cfg.heuristic_engine.rules)} heuristic rules for {dataset}")
        else:
            print("No heuristic rules loaded - continuing without heuristics")
            cfg.use_heuristics = False

    return cfg.heuristic_engine


def slugify_model_name(model_name: str) -> str:
    """Convert model name to filesystem-safe slug"""
    import re
    # Replace problematic characters with underscores
    slug = re.sub(r'[/\\:*?"<>|]', '_', model_name)
    # Replace multiple underscores with single underscore
    slug = re.sub(r'_+', '_', slug)
    # Remove leading/trailing underscores
    slug = slug.strip('_')
    # Ensure it's not empty
    if not slug:
        slug = "default_model"
    return slug


def get_embeddings_cache_path(dataset: str, model_name: str) -> pathlib.Path:
    """Get path for embeddings cache file"""
    cache_dir = pathlib.Path(".embeddings_cache")
    cache_dir.mkdir(exist_ok=True)
    safe_model_name = slugify_model_name(model_name)
    return cache_dir / f"{dataset}_{safe_model_name}_embeddings.pkl"


def compute_dataset_embeddings(dataset: str, cfg: Config) -> Dict[str, np.ndarray]:
    """Compute and cache embeddings for entire dataset"""
    cache_path = get_embeddings_cache_path(dataset, cfg.embedding_model_name)
    print(f"🔍 Looking for embeddings cache: {cache_path} (exists: {cache_path.exists()})")

    # Check if cache exists
    if cache_path.exists():
        print(f"📁 Loading cached embeddings from {cache_path}")
        try:
            with open(cache_path, "rb") as f:
                embeddings = pickle.load(f)
                print(f"✅ Loaded {len(embeddings)} cached embeddings")
                return embeddings
        except Exception as e:
            print(f"⚠️ Cache load failed: {e}, recomputing...")

    print(f"🧮 Computing embeddings for {dataset} (this may take a few minutes)...")

    model = get_semantic_model(cfg)
    if model is None:
        return {}

    # Load dataset
    root = pathlib.Path("data") / "raw" / dataset
    A_df = pd.read_csv(root / "tableA.csv")
    B_df = pd.read_csv(root / "tableB.csv")

    embeddings = {}

    # Convert records to strings and compute embeddings
    print(f"🔄 Computing embeddings for {len(A_df)} records in tableA...")
    A_strings = [json.dumps(row.to_dict(), ensure_ascii=False).lower() for _, row in A_df.iterrows()]
    A_embeddings = model.encode(A_strings, show_progress_bar=True, batch_size=32)

    print(f"🔄 Computing embeddings for {len(B_df)} records in tableB...")
    B_strings = [json.dumps(row.to_dict(), ensure_ascii=False).lower() for _, row in B_df.iterrows()]
    B_embeddings = model.encode(B_strings, show_progress_bar=True, batch_size=32)

    # Store with proper ID mapping
    if "id" in A_df.columns:
        # Use actual IDs
        for i, (_, row) in enumerate(A_df.iterrows()):
            embeddings[f"A_{row['id']}"] = A_embeddings[i]
        for i, (_, row) in enumerate(B_df.iterrows()):
            embeddings[f"B_{row['id']}"] = B_embeddings[i]
    else:
        # Use indices
        for i in range(len(A_df)):
            embeddings[f"A_{i}"] = A_embeddings[i]
        for i in range(len(B_df)):
            embeddings[f"B_{i}"] = B_embeddings[i]

    # Cache the embeddings
    print(f"💾 Caching embeddings to {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"✅ Embeddings computed and cached for {len(embeddings)} records")
    return embeddings


def semantic_similarity_cached(left_record: dict, right_record: dict, embeddings: Dict[str, np.ndarray]) -> float:
    """Calculate semantic similarity using cached embeddings"""
    try:
        # Get record IDs
        left_id = left_record.get("id", 0)
        right_id = right_record.get("id", 0)

        left_key = f"A_{left_id}"
        right_key = f"B_{right_id}"

        if left_key not in embeddings or right_key not in embeddings:
            return 0.0

        left_emb = embeddings[left_key]
        right_emb = embeddings[right_key]

        # Calculate cosine similarity
        cos_sim = np.dot(left_emb, right_emb) / (np.linalg.norm(left_emb) * np.linalg.norm(right_emb))
        return float(cos_sim)
    except Exception as e:
        print(f"Warning: Cached semantic similarity calculation failed: {e}")
        return 0.0


class CandidateCache:
    """Pre-computed cache for expensive candidate operations"""

    def __init__(self, right_records: Union[List[dict], Dict[int, dict]], cache_file: str = None):
        """Pre-compute all expensive operations on right records"""
        # Handle pandas DataFrame input
        if hasattr(right_records, 'to_dict'):
            # Convert DataFrame to dict format for compatibility
            self.right_records = right_records.set_index('id').to_dict('index')
            self.is_dict_access = True
        else:
            self.right_records = right_records
            self.is_dict_access = isinstance(right_records, dict)

        self.json_strings = {}  # id -> json string
        self.trigram_sets = {}  # id -> trigram set
        self.candidate_rankings = {}  # left_record_key -> [(candidate_id, score), ...] sorted by score desc
        self.cache_file = cache_file

        # Try to load from cache file if provided
        if cache_file and self._load_from_cache(cache_file):
            print(f"📁 Loaded candidate cache from {cache_file}")
            return

        # Build cache from scratch
        print("🔄 Building candidate cache...")
        self._build_cache()

        # Save to cache file if provided
        if cache_file:
            self._save_to_cache(cache_file)
            print(f"💾 Saved candidate cache to {cache_file}")

    def _build_cache(self):
        """Build the cache from scratch"""

        # Pre-compute JSON strings and trigram sets
        if self.is_dict_access:
            # Dict access (ID-based)
            for record_id, record in tqdm(self.right_records.items(), desc="Caching records"):
                json_str = json.dumps(record, ensure_ascii=False).lower()
                self.json_strings[record_id] = json_str
                self.trigram_sets[record_id] = self._get_trigrams(json_str)
        else:
            # List access (index-based)
            for i, record in tqdm(enumerate(self.right_records), desc="Caching records"):
                json_str = json.dumps(record, ensure_ascii=False).lower()
                self.json_strings[i] = json_str
                self.trigram_sets[i] = self._get_trigrams(json_str)

    def _save_to_cache(self, cache_file: str):
        """Save the cache to a file"""
        cache_data = {
            'json_strings': {str(k): v for k, v in self.json_strings.items()},  # Convert keys to strings for JSON
            'trigram_sets': {str(k): list(v) for k, v in self.trigram_sets.items()},  # Convert sets to lists and keys to strings for JSON
            'candidate_rankings': self.candidate_rankings,  # Cache the computed rankings
            'is_dict_access': self.is_dict_access
        }

        cache_path = pathlib.Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)

    def _load_from_cache(self, cache_file: str) -> bool:
        """Load the cache from a file. Returns True if successful."""
        cache_path = pathlib.Path(cache_file)
        if not cache_path.exists():
            return False

        # Check if cache is recent (less than 7 days old)
        try:
            file_age = time.time() - cache_path.stat().st_mtime
            if file_age > 604800:  # 7 days in seconds (7 * 24 * 3600)
                print(f"⚠️ Cache file {cache_file} is {file_age/86400:.1f} days old, rebuilding...")
                return False
        except Exception:
            return False

        try:
            with open(cache_file) as f:
                cache_data = json.load(f)

            self.json_strings = cache_data['json_strings']
            self.trigram_sets = {k: set(v) for k, v in cache_data['trigram_sets'].items()}  # Convert lists back to sets
            # Load rankings cache and convert lists back to tuples (JSON serializes tuples as lists)
            raw_rankings = cache_data.get('candidate_rankings', {})
            self.candidate_rankings = {
                k: [(idx, score) for idx, score in rankings]  # Convert [idx, score] back to (idx, score)
                for k, rankings in raw_rankings.items()
            }
            self.is_dict_access = cache_data['is_dict_access']

            # Convert string keys back to int for list access
            if not self.is_dict_access:
                self.json_strings = {int(k): v for k, v in self.json_strings.items()}
                self.trigram_sets = {int(k): v for k, v in self.trigram_sets.items()}
            # For dict access with integer keys, convert string keys back to ints
            # Sample a key to determine the original type
            elif self.right_records:
                sample_key = next(iter(self.right_records.keys()))
                if isinstance(sample_key, int):
                    # Convert all string keys back to integers
                    self.json_strings = {int(k): v for k, v in self.json_strings.items()}
                    self.trigram_sets = {int(k): v for k, v in self.trigram_sets.items()}

            return True
        except Exception as e:
            print(f"⚠️ Failed to load cache from {cache_file}: {e}")
            return False

        print(f"✅ Cached {len(self.json_strings)} records and {len(self.candidate_rankings)} ranking sets")
        return None

    def _generate_rankings_cache_key(self, left_record: dict, weights: Tuple[float, float, float]) -> str:
        """Generate cache key for candidate rankings based on record content and weights"""
        import hashlib
        
        # Create deterministic string from record (sorted to ensure consistency)
        record_str = json.dumps(left_record, sort_keys=True, ensure_ascii=False)
        
        # Include weights in key so cache invalidates when weights change
        semantic_weight, trigram_weight, syntactic_weight = weights
        weights_str = f"{semantic_weight:.6f}_{trigram_weight:.6f}_{syntactic_weight:.6f}"
        
        # Create hash for efficient key storage
        combined_str = f"{record_str}|{weights_str}"
        return hashlib.md5(combined_str.encode('utf-8')).hexdigest()

    def get_cache_stats(self) -> dict:
        """Get cache statistics for debugging"""
        return {
            "preprocessed_records": len(self.json_strings),
            "cached_ranking_sets": len(self.candidate_rankings),
            "total_cached_rankings": sum(len(rankings) for rankings in self.candidate_rankings.values())
        }

    def _get_trigrams(self, s: str) -> set:
        """Get trigram set for a string"""
        if len(s) < 3:
            return {s}
        return {s[i : i + 3] for i in range(len(s) - 2)}

    def compute_trigram_similarity(self, left_str: str, right_id: Union[int, str]) -> float:
        """Fast trigram similarity using pre-computed trigrams"""
        left_trigrams = self._get_trigrams(left_str)
        right_trigrams = self.trigram_sets[right_id]

        if not left_trigrams and not right_trigrams:
            return 1.0
        if not left_trigrams or not right_trigrams:
            return 0.0

        intersection = len(left_trigrams & right_trigrams)
        union = len(left_trigrams | right_trigrams)
        return intersection / union if union > 0 else 0.0

    def compute_syntactic_similarity(self, left_str: str, right_id: Union[int, str]) -> float:
        """Fast syntactic similarity using pre-computed JSON strings"""
        right_str = self.json_strings[right_id]
        return syntactic_similarity(left_str, right_str)

    def compute_semantic_similarity(self, left_record: dict, right_id: Union[int, str], embeddings: Dict[str, np.ndarray]) -> float:
        """Fast semantic similarity using cached embeddings"""
        if embeddings is None:
            return 0.0
        right_record = self.get_record(right_id)
        return semantic_similarity_cached(left_record, right_record, embeddings)

    def compute_combined_similarity(self, left_record: dict, left_str: str, right_id: Union[int, str], cfg: Config) -> float:
        """Compute the full 3-weight combined similarity score"""
        # Calculate all three similarity scores
        trigram_score = self.compute_trigram_similarity(left_str, right_id)
        syntactic_score = self.compute_syntactic_similarity(left_str, right_id)

        # Handle semantic similarity availability
        if cfg.use_semantic and cfg.embeddings is not None:
            semantic_score = self.compute_semantic_similarity(left_record, right_id, cfg.embeddings)
        else:
            semantic_score = 0.0
            # If semantic unavailable, redistribute its weight to trigram and syntactic
            if cfg.semantic_weight > 0.0:
                total_non_semantic = cfg.trigram_weight + cfg.syntactic_weight
                if total_non_semantic > 0.0:
                    trigram_weight_adj = cfg.trigram_weight + (cfg.semantic_weight * cfg.trigram_weight / total_non_semantic)
                    syntactic_weight_adj = cfg.syntactic_weight + (cfg.semantic_weight * cfg.syntactic_weight / total_non_semantic)
                    return trigram_weight_adj * trigram_score + syntactic_weight_adj * syntactic_score

        # 3-weight combination
        return (cfg.trigram_weight * trigram_score +
                cfg.syntactic_weight * syntactic_score +
                cfg.semantic_weight * semantic_score)

    def get_record(self, record_id: Union[int, str]) -> dict:
        """Get record by ID"""
        if self.is_dict_access:
            return self.right_records[record_id]
        return self.right_records[record_id]

    def get_json_string(self, record_id: Union[int, str]) -> str:
        """Get pre-computed JSON string"""
        return self.json_strings[record_id]

    def get_all_ids(self) -> List[Union[int, str]]:
        """Get all record IDs"""
        if self.is_dict_access:
            return list(self.right_records.keys())
        return list(range(len(self.right_records)))


def get_top_candidates_cached(
    left_record: dict, candidate_cache: CandidateCache, max_candidates: int, cfg: Config, dataset: str = None
) -> List[tuple]:
    """FAST get_top_candidates using pre-computed rankings cache - computes similarities once, caches forever"""
    
    # Generate cache key based on record content and weights
    weights = (cfg.semantic_weight, cfg.trigram_weight, cfg.syntactic_weight)
    cache_key = candidate_cache._generate_rankings_cache_key(left_record, weights)
    
    # Check if we have cached rankings for this record+weights combination
    if cache_key in candidate_cache.candidate_rankings:
        # Return top N from cached full rankings
        cached_rankings = candidate_cache.candidate_rankings[cache_key]
        return [(idx, candidate_cache.get_record(idx)) for idx, _ in cached_rankings[:max_candidates]]
    
    # Cache miss - compute similarities for ALL candidates and cache the full rankings
    left_str = json.dumps(left_record, ensure_ascii=False).lower()
    heuristic_engine = get_heuristic_engine(cfg, dataset) if dataset else None
    
    # Compute similarity scores for all candidates
    all_similarities = []
    
    # Add progress bar for expensive computation
    all_ids = list(candidate_cache.get_all_ids())
    try:
        from tqdm import tqdm
        id_iterator = tqdm(all_ids, desc="Computing candidate similarities", unit="candidates")
    except ImportError:
        id_iterator = all_ids
    
    for record_id in id_iterator:
        # Use the combined similarity method that computes all 3 types
        combined_score = candidate_cache.compute_combined_similarity(left_record, left_str, record_id, cfg)
        record = candidate_cache.get_record(record_id)

        # Apply heuristics if enabled
        if heuristic_engine:
            try:
                # Candidate generation heuristics
                candidate_action = heuristic_engine.apply_stage_heuristics(
                    "candidate_generation", left_record, record
                )
                if candidate_action and hasattr(candidate_action, "similarity_boost"):
                    combined_score += candidate_action.similarity_boost * candidate_action.confidence
                    combined_score = min(combined_score, 1.0)

                # Other heuristics
                heuristic_adjustment = heuristic_engine.apply_heuristics(left_record, record)
                combined_score += heuristic_adjustment
            except Exception:
                pass

        all_similarities.append((record_id, combined_score))

    # Sort by score (descending) and cache the FULL rankings
    all_similarities.sort(key=lambda x: x[1], reverse=True)
    candidate_cache.candidate_rankings[cache_key] = all_similarities
    
    # Save updated cache to disk if cache file is specified
    if candidate_cache.cache_file:
        try:
            candidate_cache._save_to_cache(candidate_cache.cache_file)
        except Exception as e:
            print(f"⚠️ Failed to update candidate cache file: {e}")
    
    # Return top N requested
    return [(idx, candidate_cache.get_record(idx)) for idx, _ in all_similarities[:max_candidates]]


def create_candidate_cache(dataset: str, B: dict, cfg: Config, max_candidates: int) -> CandidateCache:
    """
    Create or reuse candidate cache with smart cache file selection.

    Args:
        dataset: Dataset name
        B: Table B records
        cfg: Configuration object with embedding model info
        max_candidates: Maximum candidates needed

    Returns:
        CandidateCache: Ready-to-use candidate cache
    """
    print("🔄 Creating candidate cache...")

    # Include model name in cache to avoid conflicts between different embedding models
    safe_model_name = slugify_model_name(cfg.embedding_model_name)
    cache_file = f".candidate_cache/{dataset}_{safe_model_name}_candidates_{max_candidates}.json"

    # Try to find a larger cache file we can use
    cache_dir = pathlib.Path(".candidate_cache")
    cache_pattern = f"{dataset}_{safe_model_name}_candidates_*.json"
    existing_caches = list(cache_dir.glob(cache_pattern))

    # Find the largest cache that has >= max_candidates
    best_cache_file = cache_file
    best_cache_size = max_candidates

    for existing_cache in existing_caches:
        # Extract candidates count from filename
        try:
            cache_candidates = int(existing_cache.stem.split('_candidates_')[1])
            if cache_candidates >= max_candidates and cache_candidates > best_cache_size:
                best_cache_file = str(existing_cache)
                best_cache_size = cache_candidates
        except (ValueError, IndexError):
            continue

    if best_cache_file != cache_file:
        print(f"🔍 Found larger cache with {best_cache_size} candidates, reusing: {best_cache_file}")

    candidate_cache = CandidateCache(B, cache_file=best_cache_file)
    print(f"✅ Candidate cache created for {len(B)} records")

    return candidate_cache


def load_dataset_and_pairs(dataset: str, use_validation: bool = False):
    """
    Load dataset tables and pairs with proper ID mapping and validation handling.

    Returns:
        tuple: (A, B, pairs) where A and B are either dicts (ID mapping) or lists (indexing)
    """
    # Load data with proper ID mapping
    root = pathlib.Path("data") / "raw" / dataset
    A_df = pd.read_csv(root / "tableA.csv")
    B_df = pd.read_csv(root / "tableB.csv")

    # Check if this dataset has non-sequential IDs (like zomato_yelp)
    if "id" in A_df.columns:
        # Create ID-to-record mappings
        A = {row["id"]: row.to_dict() for _, row in A_df.iterrows()}
        B = {row["id"]: row.to_dict() for _, row in B_df.iterrows()}
        print(f"Dataset uses ID mapping: A has {len(A)} records (IDs {min(A.keys())}-{max(A.keys())})")
    else:
        # Use list indexing for datasets without ID column
        A = A_df.to_dict(orient="records")
        B = B_df.to_dict(orient="records")
        print(f"Dataset uses list indexing: A has {len(A)} records")

    # Load pairs data - use validation data if flag is set to prevent data leakage
    if use_validation:
        if (root / "valid.csv").exists():
            all_pairs = pd.read_csv(root / "valid.csv")
            # Use balanced sampling to keep evaluation manageable (~200 pairs)
            from src.entity_matching.balanced_sampling import balanced_train_sample, get_sample_info
            pairs = balanced_train_sample(all_pairs, target_size=200, random_state=42)
            print("✅ Using validation set for evaluation (200-pair balanced sample, no test data leakage)")
            print(get_sample_info(pairs, "validation sample"))
        elif (root / "train.csv").exists():
            train_pairs = pd.read_csv(root / "train.csv")

            # Ensure balanced sampling to avoid all-positive or all-negative slices
            positive_pairs = train_pairs[train_pairs['label'] == 1]
            negative_pairs = train_pairs[train_pairs['label'] == 0]

            # Take up to 50 of each class for balanced evaluation (100 total max)
            max_per_class = 50
            pos_sample = positive_pairs.head(min(max_per_class, len(positive_pairs)))
            neg_sample = negative_pairs.head(min(max_per_class, len(negative_pairs)))

            # Combine and shuffle
            pairs = pd.concat([pos_sample, neg_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

            print(f"✅ Using balanced train sample: {len(pos_sample)} positive + {len(neg_sample)} negative = {len(pairs)} total pairs (no test data leakage)")
        else:
            raise ValueError("No validation or training data available - cannot evaluate without test data leakage")
    else:
        pairs = pd.read_csv(root / "test.csv")
        print("⚠️ Using test set for evaluation")

    return A, B, pairs


async def enhanced_match_single_record(
    left_record: dict,
    candidates: List[tuple],
    cfg: Config,
    client: AsyncOpenAI,
    heuristic_engine: EnhancedHeuristicEngine,
    candidate_cache: CandidateCache,
    prompt_data: Optional[Dict[str, Any]] = None,
) -> tuple[int, bool]:
    """Enhanced matching with sophisticated control logic"""
    logger = logging.getLogger(__name__)

    # Early decision check before LLM
    best_candidate = None
    best_score = 0.0
    weight_adjustments = 0
    left_str = json.dumps(left_record, ensure_ascii=False).lower()

    for idx, candidate_record in candidates:
        # Use cache's combined similarity calculation (handles all 3 similarity types properly)
        combined_score = candidate_cache.compute_combined_similarity(left_record, left_str, idx, cfg)

        # Apply weight rules to potentially adjust semantic weight
        current_weights = {"semantic_weight": cfg.semantic_weight}
        weight_action = heuristic_engine.apply_weight_rules(
            left_record, candidate_record, current_weights, PipelineStage.PRE_SEMANTIC
        )

        # If heuristics want to adjust weights, recalculate with adjusted weights
        if weight_action and weight_action.semantic_weight is not None:
            effective_semantic_weight = weight_action.semantic_weight
            weight_adjustments += 1

            # Recalculate with adjusted semantic weight
            trigram_score = candidate_cache.compute_trigram_similarity(left_str, idx)
            syntactic_score = candidate_cache.compute_syntactic_similarity(left_str, idx)
            semantic_score = candidate_cache.compute_semantic_similarity(left_record, idx, cfg.embeddings)

            # Apply adjusted weights (note: this is a simplified 2-weight system for heuristic adjustments)
            combined_score = (1 - effective_semantic_weight) * ((trigram_score + syntactic_score) / 2) + effective_semantic_weight * semantic_score

        # Apply score rules
        score_adjustment = heuristic_engine.apply_score_rules(
            left_record, candidate_record, PipelineStage.CANDIDATE_SELECTION
        )
        final_score = combined_score + score_adjustment

        if final_score > best_score:
            best_score = final_score
            best_candidate = (idx, candidate_record)

    if not best_candidate:
        return -1, False

    best_idx, best_record = best_candidate

    # Summarize weight adjustments to reduce noise
    if weight_adjustments > 0:
        pass  # Applied weight adjustments (removed verbose logging)

    # Apply decision rules before LLM call
    decision = heuristic_engine.apply_decision_rules(left_record, best_record, best_score, PipelineStage.PRE_LLM)

    if decision and decision.terminate_early:
        if decision.skip_llm:
            pass  # Early decision made (removed verbose logging)
        else:
            pass  # Early decision made (removed verbose logging)
        # If rule says accept (1), return the best candidate index; if reject (0), return -1
        return (best_idx if decision.final_result == 1 else -1), True

    # Fall back to LLM if no early decision
    # Proceeding to LLM (removed verbose logging)

    # Build prompt with all candidates (recall@N) - fixes recall@1 architecture flaw
    candidates_lines = []
    for position, (actual_id, record) in enumerate(candidates, 1):
        # Remove the 'id' field from the record to prevent LLM confusion (match baseline behavior)
        cleaned_record = {k: v for k, v in record.items() if k != 'id'}
        candidates_lines.append(f"{position}) {json.dumps(cleaned_record, ensure_ascii=False)}")
    candidates_text = "\n".join(candidates_lines)

    # best_idx is actually a record ID, find its position in the candidates list
    display_best_idx = 1  # Default to first candidate if not found
    for position, (actual_id, record) in enumerate(candidates, 1):
        if actual_id == best_idx:
            display_best_idx = position
            break

    # DEBUG LOGGING: Log candidate display info
    logger.debug(f"🔍 Candidate mapping: best_idx (record_id)={best_idx}, display_best_idx (position)={display_best_idx}")
    logger.debug(f"🔍 Showing {len(candidates)} candidates to LLM, positions 1-{len(candidates)}")

    # Format the complete prompt using structured sections
    prompt = build_prompt(
        left_record=left_record,
        candidates_text=candidates_text,
        best_idx=display_best_idx,
        prompt_data=prompt_data,
        additional_guidance=None
    )

    # Check token count
    total_tokens = token_count(prompt, cfg.model)
    if total_tokens > 1000000:  # 1M token limit
        print(f"  WARNING: Prompt too large ({total_tokens:,} tokens)")
        return -1, False

    # Get LLM response
    response = await call_openai_async(prompt, cfg, client)

    # DEBUG LOGGING: Log LLM interaction
    logger.debug(f"🔍 LLM raw response: '{response}'")

    # Parse response
    if not response:
        print("  WARNING: Empty response from LLM")
        return -1, False

    try:
        # Clean the response - remove whitespace and try to extract number
        response_clean = response.strip()

        # Try direct int conversion first
        try:
            llm_choice = int(response_clean)
        except ValueError:
            # Try to extract number from response if it contains extra text
            numbers = re.findall(r'-?\d+', response_clean)
            if numbers:
                llm_choice = int(numbers[0])  # Take first number found
                print(f"  DEBUG: Extracted number {llm_choice} from response: '{response_clean}'")
            else:
                print(f"  WARNING: Could not parse LLM response: '{response_clean}' (no numbers found)")
                return -1, False

        # Handle LLM response: -1 means no match, 1-N means candidate choice
        if llm_choice == -1:
            logger.debug("🔍 LLM chose no match (-1)")
            return -1, False
        if 1 <= llm_choice <= len(candidates):
            # Convert 1-based LLM response to 0-based candidate index
            candidate_idx = llm_choice - 1
            chosen_right_id, _ = candidates[candidate_idx]
            logger.debug(f"🔍 LLM chose position {llm_choice} → candidate_idx={candidate_idx} → right_id={chosen_right_id}")
            # Return the candidate index (not the ID - calling code expects index)
            return candidate_idx, False
        logger.warning(f"🔍 LLM chose invalid candidate {llm_choice} (valid range: 1-{len(candidates)} or -1)")
        return -1, False

    except Exception as e:
        print(f"  WARNING: Error parsing LLM response: '{response}', error: {e}")
        return -1, False


async def run_enhanced_matching(
    experiment: ExperimentConfig,
) -> Dict:
    """Run enhanced entity matching with sophisticated control logic"""

    dataset = experiment.dataset
    model = experiment.llm_model
    max_candidates = experiment.max_candidates
    heuristic_file = experiment.heuristic_file
    use_validation = experiment.use_validation
    embedding_base_url = experiment.embedding_base_url
    embedding_model = experiment.embedding_model
    semantic_weight = experiment.semantic_weight
    trigram_weight = experiment.trigram_weight
    syntactic_weight = experiment.syntactic_weight
    concurrency = experiment.concurrency

    print("🚀 ENHANCED ENTITY MATCHING")
    print(f"Dataset: {dataset}")
    print(f"Model: {model}")
    print(f"Candidates: {max_candidates}")
    print(f"Heuristics: {heuristic_file}")
    print("=" * 80)

    # Load enhanced heuristic engine
    heuristic_engine = load_enhanced_heuristics_for_dataset(dataset, heuristic_file)

    # # Load configuration from heuristic file if provided
    # prompt_data = None
    # if heuristic_file and os.path.exists(heuristic_file):
    #     try:
    #         with open(heuristic_file) as f:
    #             heuristic_config = json.load(f)

    #         if "hyperparameters" in heuristic_config:
    #             hyperparams = heuristic_config["hyperparameters"]

    #             # Override weights from heuristic file
    #             file_semantic = hyperparams.get("semantic_weight")
    #             file_trigram = hyperparams.get("trigram_weight")
    #             file_syntactic = hyperparams.get("syntactic_weight")

    #             if file_semantic is not None:
    #                 semantic_weight = file_semantic
    #             if file_trigram is not None:
    #                 trigram_weight = file_trigram
    #             if file_syntactic is not None:
    #                 syntactic_weight = file_syntactic

    #             print(f"✅ Override weights from heuristic file: semantic={semantic_weight}, trigram={trigram_weight}, syntactic={syntactic_weight} (overriding previous values)")
    #         else:
    #             print("⚠️ No hyperparameters found in heuristic file")

    #         # Load prompt data from heuristic file if available
    #         if "prompt_data" in heuristic_config:
    #             prompt_data = heuristic_config["prompt_data"]
    #             print("✅ Loaded prompt data from heuristic file")

    #     except Exception as e:
    #         print(f"⚠️ Could not extract configuration from heuristic file: {e}")

    print(f"🎯 FINAL weights being used: semantic={semantic_weight}, trigram={trigram_weight}, syntactic={syntactic_weight}")
    print("=" * 80)

    prompt_data = get_prompt_data()

    # Initialize configuration
    cfg = Config()
    cfg.model = model
    cfg.use_semantic = True

    # Set embedding configuration
    if embedding_base_url:
        cfg.embedding_base_url = embedding_base_url
        cfg.embedding_model_name = embedding_model
        print(f"🤖 Using embedding API: {embedding_base_url} with model {embedding_model}")
    else:
        cfg.embedding_model_name = embedding_model
        print(f"🤖 Using local embedding model: {embedding_model}")

    # Always use 3-weight system (legacy 2-weight system removed)
    cfg.set_weights(trigram_weight, syntactic_weight, semantic_weight)

    # Initialize embeddings cache if using semantic similarity
    if cfg.use_semantic:
        cfg.embeddings = compute_dataset_embeddings(dataset, cfg)

    # Initialize async OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Load data with proper ID mapping
    A, B, pairs = load_dataset_and_pairs(dataset, use_validation)

    # Note: limit parameter removed - always use full evaluation set for reliable results

    print(f"Processing {len(pairs)} pairs with {max_candidates} candidates ({max_candidates / len(B):.1%} of table B) per record")

    # Create candidate cache for massive speed improvement with persistent caching
    candidate_cache = create_candidate_cache(dataset, B, cfg, max_candidates)

    # Embeddings cache already initialized in run_enhanced_matching function

    start_time = time.time()
    all_predictions = {}
    early_decisions = 0
    llm_calls = 0
    early_decision_pairs = set()  # Track which pairs had early decisions

    # Process pairs with async concurrency
    async def process_single_pair(row):
        """Process a single pair with enhanced matching"""
        left_id = row.ltable_id
        left_record = A[left_id]

        # Use cached candidate generation for massive speed improvement
        top_candidates = get_top_candidates_cached(
            left_record, candidate_cache, max_candidates, cfg, dataset
        )

        # Enhanced matching with control logic
        match_idx, was_early_decision = await enhanced_match_single_record(
            left_record, top_candidates, cfg, client, heuristic_engine, candidate_cache, prompt_data
        )

        # Return results for thread-safe aggregation (include candidates for ID mapping)
        return left_id, match_idx, was_early_decision, top_candidates

    # Create batches for concurrent processing using dynamic batching like baseline
    batch_size = max(1, len(pairs) // 20)  # Dynamic batch size for ~20 progress updates
    pair_rows = list(pairs.iterrows())

    # Create batches
    batches = []
    for i in range(0, len(pair_rows), batch_size):
        batches.append(pair_rows[i : i + batch_size])

    # Use semaphore for concurrency control like baseline
    semaphore = asyncio.Semaphore(concurrency)

    async def process_batch_with_semaphore(batch):
        """Process a batch of pairs with semaphore control"""
        async with semaphore:
            tasks = [process_single_pair(row) for _, row in batch]
            return await asyncio.gather(*tasks)

    # Process batches with progress tracking
    with tqdm(total=len(batches), desc="Processing batches", unit="batch") as pbar:
        # Process all batches concurrently (limited by semaphore)
        batch_tasks = [process_batch_with_semaphore(batch) for batch in batches]

        for task in asyncio.as_completed(batch_tasks):
            batch_results = await task

            # Aggregate results thread-safely
            for left_id, match_idx, was_early_decision, top_candidates in batch_results:
                if match_idx != -1:
                    # match_idx is a candidate index, convert to right_id for evaluation
                    right_id, _ = top_candidates[match_idx]
                    all_predictions[left_id] = right_id

                if was_early_decision:
                    early_decisions += 1
                    early_decision_pairs.add(left_id)
                else:
                    llm_calls += 1

            pbar.update(1)  # Update by 1 batch completed

    elapsed_time = time.time() - start_time
    matches_found = len(all_predictions)

    print("\n=== ENHANCED MATCHING RESULTS ===")
    print(f"Processed: {len(pairs)} pairs")
    print(f"Matches found: {matches_found}")
    print(f"Early decisions: {early_decisions}")
    print(f"LLM calls: {llm_calls}")
    print(f"LLM call reduction: {(1 - llm_calls / len(pairs)) * 100:.1f}%")
    print(f"Processing time: {elapsed_time:.1f} seconds")

    # B is already in the right format - either dict or list of dicts
    if isinstance(B, dict):
        # Dict case (ID -> record)
        B_dict = B
    else:
        # List of dicts case
        B_dict = {r['id']: r for r in B}

    # Prepare pairs data for evaluation
    pairs_data = [(rec.ltable_id, rec.rtable_id, rec.label) for _, rec in pairs.iterrows()]

    # Use detailed duplicate-aware evaluation
    detailed_results = duplicate_aware_detailed_evaluate(all_predictions, pairs_data, A, B_dict, verbose=True)
    preds, labels = detailed_results["preds"], detailed_results["labels"]

    # Calculate metrics using shared function
    metrics = calculate_metrics(preds, labels)
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    precision, recall, f1, accuracy = metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]

    # Calculate cost
    try:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS[cfg.model]
    except KeyError:
        input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get("gpt-4o-mini", (0.00015, 0.0006))

    input_cost = (cfg.total_input_tokens / 1_000_000) * input_cost_per_1k
    output_cost = (cfg.total_output_tokens / 1_000_000) * output_cost_per_1k
    total_cost = input_cost + output_cost

    print("\n=== PERFORMANCE METRICS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"Cost: ${total_cost:.4f}")
    print(f"Candidate selection: {max_candidates} candidates ({max_candidates / len(B):.1%} of table B)")
    print(f"Tokens: {cfg.total_input_tokens} input, {cfg.total_output_tokens} output")

    # Add similarity calculations to the detailed results for failure analysis
    false_positives = detailed_results["false_positives"]
    false_negatives = detailed_results["false_negatives"]
    true_positives = detailed_results["true_positives"]

    # Enhance false positives with similarity calculations
    for fp in false_positives:
        if fp["predicted_record"]:
            left_str = json.dumps(fp["left_record"], ensure_ascii=False).lower()
            pred_str = json.dumps(fp["predicted_record"], ensure_ascii=False).lower()
            actual_str = json.dumps(fp["actual_record"], ensure_ascii=False).lower()

            fp["predicted_similarity"] = {
                "trigram": trigram_similarity(left_str, pred_str),
                "syntactic": syntactic_similarity(left_str, pred_str),
                "semantic": semantic_similarity_cached(fp["left_record"], fp["predicted_record"], cfg.embeddings) if cfg.use_semantic else 0.0
            }
            fp["actual_similarity"] = {
                "trigram": trigram_similarity(left_str, actual_str),
                "syntactic": syntactic_similarity(left_str, actual_str),
                "semantic": semantic_similarity_cached(fp["left_record"], fp["actual_record"], cfg.embeddings) if cfg.use_semantic else 0.0
            }

    # Enhance false negatives with similarity and candidate analysis
    for fn in false_negatives:
        left_str = json.dumps(fn["left_record"], ensure_ascii=False).lower()
        missed_str = json.dumps(fn["missed_record"], ensure_ascii=False).lower()

        fn["similarity"] = {
            "trigram": trigram_similarity(left_str, missed_str),
            "syntactic": syntactic_similarity(left_str, missed_str),
            "semantic": semantic_similarity_cached(fn["left_record"], fn["missed_record"], cfg.embeddings) if cfg.use_semantic else 0.0
        }

        # Check if actual match was in candidates
        candidates = get_top_candidates_cached(fn["left_record"], candidate_cache, max_candidates, cfg, dataset)
        found_in_candidates = any(idx == fn["true_right_id"] for idx, _ in candidates)
        candidate_rank = None
        if found_in_candidates:
            for rank, (idx, _) in enumerate(candidates, 1):
                if idx == fn["true_right_id"]:
                    candidate_rank = rank
                    break

        fn["candidate_analysis"] = {
            "found_in_candidates": found_in_candidates,
            "rank": candidate_rank,
            "max_candidates": max_candidates
        }

    # Enhance true positives with similarity calculations
    for tp in true_positives:
        left_str = json.dumps(tp["left_record"], ensure_ascii=False).lower()
        matched_str = json.dumps(tp["matched_record"], ensure_ascii=False).lower()

        tp["similarity"] = {
            "trigram": trigram_similarity(left_str, matched_str),
            "syntactic": syntactic_similarity(left_str, matched_str),
            "semantic": semantic_similarity_cached(tp["left_record"], tp["matched_record"], cfg.embeddings) if cfg.use_semantic else 0.0
        }

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "cost": total_cost,
        "early_decisions": early_decisions,
        "llm_calls": llm_calls,
        "llm_call_reduction": (1 - llm_calls / len(pairs)) * 100,
        "predictions": all_predictions,  # Include predictions for failure analysis
        "failure_analysis": {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_positives": true_positives,
            "true_negatives": detailed_results["true_negatives"],
            "summary": detailed_results["summary"]
        }
    }
