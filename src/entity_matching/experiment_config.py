#!/usr/bin/env python
"""
Comprehensive experiment configuration system.

This module provides a complete, serializable configuration class that captures
all experiment settings in one place, enabling:
- Consistent configuration between MCP server and main pipeline
- Complete experiment reproducibility
- Proper experiment tracking and sharing
"""

import json
import uuid

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .hybrid_matcher import Config


@dataclass
class ExperimentConfig:
    """Complete experiment configuration with serialization support"""

    # Experiment metadata
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"

    # Dataset settings
    dataset: str = None
    use_validation: bool = False

    # Model settings
    llm_model: str = "gpt-4.1-nano"
    temperature: float = 0
    max_tokens: int = 100

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_base_url: Optional[str] = None

    # Weight system
    semantic_weight: float = 0.5
    trigram_weight: Optional[float] = None  # Auto-calculated if None
    syntactic_weight: Optional[float] = None  # Auto-calculated if None

    # Candidate settings
    max_candidates: int = 50

    # Pipeline settings
    concurrency: int = 10
    mode: str = "prompt-modification"  # or "heuristics"

    # Heuristics
    heuristic_file: Optional[str] = None
    use_train_for_rules: bool = False

    # Prompt customization
    prompt_data: Optional[Dict] = None

    # Cache settings
    no_cache: bool = False

    # Runtime state (not serialized)
    _semantic_model: Any = field(default=None, repr=False, init=False)
    _embeddings: Dict = field(default=None, repr=False, init=False)
    _total_cost: float = field(default=0.0, repr=False, init=False)

    def __post_init__(self):
        """Auto-calculate missing weights if needed"""
        if self.trigram_weight is None or self.syntactic_weight is None:
            remaining_weight = 1.0 - self.semantic_weight
            self.trigram_weight = remaining_weight * 0.6  # 60% of remaining
            self.syntactic_weight = remaining_weight * 0.4  # 40% of remaining

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding runtime state"""
        config_dict = asdict(self)
        # Remove runtime fields that start with underscore
        return {k: v for k, v in config_dict.items() if not k.startswith('_')}

    def to_file(self, path: str) -> None:
        """Save complete config to JSON file"""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        # Filter out any keys that aren't in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_file(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file"""
        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_legacy_config(self) -> Config:
        """Convert to legacy Config class for backward compatibility"""
        cfg = Config()

        # Model settings
        cfg.model = self.llm_model
        cfg.temperature = self.temperature
        cfg.max_tokens = self.max_tokens

        # Embedding settings
        cfg.embedding_model_name = self.embedding_model
        cfg.embedding_base_url = self.embedding_base_url

        # Weight system
        cfg.set_weights(self.trigram_weight, self.syntactic_weight, self.semantic_weight)

        # Heuristics
        cfg.heuristic_file = self.heuristic_file
        cfg.use_heuristics = bool(self.heuristic_file)

        # Copy runtime state if present
        if hasattr(self, '_semantic_model'):
            cfg.semantic_model = self._semantic_model
        if hasattr(self, '_embeddings'):
            cfg.embeddings = self._embeddings
        if hasattr(self, '_total_cost'):
            cfg.total_cost = self._total_cost

        return cfg

    @classmethod
    def from_args(cls, args) -> 'ExperimentConfig':
        """Create config from argparse namespace"""
        config = cls()

        # Map argparse arguments to config fields
        if hasattr(args, 'dataset') and args.dataset:
            config.dataset = args.dataset
        if hasattr(args, 'model') and args.model:
            config.llm_model = args.model
        if hasattr(args, 'embedding_model') and args.embedding_model:
            config.embedding_model = args.embedding_model
        if hasattr(args, 'embedding_base_url') and args.embedding_base_url:
            config.embedding_base_url = args.embedding_base_url
        if hasattr(args, 'max_candidates') and args.max_candidates:
            config.max_candidates = args.max_candidates
        if hasattr(args, 'semantic_weight') and args.semantic_weight is not None:
            config.semantic_weight = args.semantic_weight
        if hasattr(args, 'trigram_weight') and args.trigram_weight is not None:
            config.trigram_weight = args.trigram_weight
        if hasattr(args, 'syntactic_weight') and args.syntactic_weight is not None:
            config.syntactic_weight = args.syntactic_weight
        if hasattr(args, 'concurrency') and args.concurrency:
            config.concurrency = args.concurrency
        if hasattr(args, 'heuristic_file') and args.heuristic_file:
            config.heuristic_file = args.heuristic_file
        if hasattr(args, 'use_validation') and args.use_validation:
            config.use_validation = args.use_validation
        if hasattr(args, 'mode') and args.mode:
            config.mode = args.mode
        if hasattr(args, 'no_cache') and args.no_cache:
            config.no_cache = args.no_cache

        return config

    def validate(self) -> None:
        """Validate configuration consistency"""
        if not self.dataset:
            raise ValueError("Dataset must be specified")

        if self.semantic_weight < 0 or self.semantic_weight > 1:
            raise ValueError(f"Semantic weight must be 0-1, got {self.semantic_weight}")

        if self.trigram_weight is not None and (self.trigram_weight < 0 or self.trigram_weight > 1):
            raise ValueError(f"Trigram weight must be 0-1, got {self.trigram_weight}")

        if self.syntactic_weight is not None and (self.syntactic_weight < 0 or self.syntactic_weight > 1):
            raise ValueError(f"Syntactic weight must be 0-1, got {self.syntactic_weight}")

        # Check weights sum to 1.0 (within tolerance)
        total_weight = self.semantic_weight + (self.trigram_weight or 0) + (self.syntactic_weight or 0)
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        if self.max_candidates <= 0:
            raise ValueError(f"Max candidates must be positive, got {self.max_candidates}")

        if self.concurrency <= 0:
            raise ValueError(f"Concurrency must be positive, got {self.concurrency}")

    def get_cache_key(self) -> str:
        """Generate cache key based on relevant settings"""
        from .hybrid_matcher import slugify_model_name

        safe_model_name = slugify_model_name(self.embedding_model)
        return f"{self.dataset}_{safe_model_name}_candidates_{self.max_candidates}"

    def get_experiment_dir(self, base_dir: str = "results/experiments") -> Path:
        """Get directory for this experiment's files"""
        exp_dir = Path(base_dir) / f"exp_{self.experiment_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save_experiment(self, base_dir: str = "results/experiments") -> Path:
        """Save complete experiment configuration"""
        exp_dir = self.get_experiment_dir(base_dir)
        config_path = exp_dir / "config.json"
        self.to_file(str(config_path))
        return config_path

    def __str__(self) -> str:
        """Human-readable configuration summary"""
        return (
            f"ExperimentConfig(id={self.experiment_id}, dataset={self.dataset}, "
            f"model={self.llm_model}, embedding={self.embedding_model}, "
            f"weights={self.semantic_weight:.2f}/{self.trigram_weight:.2f}/{self.syntactic_weight:.2f}, "
            f"candidates={self.max_candidates})"
        )


def create_config_from_legacy(cfg: Config, **overrides) -> ExperimentConfig:
    """Create ExperimentConfig from legacy Config object"""
    config = ExperimentConfig(
        llm_model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        embedding_model=cfg.embedding_model_name,
        embedding_base_url=cfg.embedding_base_url,
        semantic_weight=cfg.semantic_weight,
        trigram_weight=cfg.trigram_weight,
        syntactic_weight=cfg.syntactic_weight,
        heuristic_file=cfg.heuristic_file,
        **overrides
    )

    # Copy runtime state
    config._semantic_model = cfg.semantic_model
    config._embeddings = cfg.embeddings
    config._total_cost = cfg.total_cost

    return config
