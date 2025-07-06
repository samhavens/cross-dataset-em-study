#!/usr/bin/env python
"""
Parse dataset sizes and leaderboard information for automated experiments.
"""

import re
import pathlib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DatasetSizes:
    """Dataset size and token information"""
    dataset: str
    table_a_records: int
    table_b_records: int
    total_records: int
    avg_tokens_a: float
    avg_tokens_b: float
    test_pairs: int
    positive_pairs: int
    negative_pairs: int
    positive_rate: float

@dataclass
class LeaderboardEntry:
    """Single leaderboard entry"""
    rank: int
    model: str
    f1: float
    is_best: bool
    is_jellyfish: bool  # Seen during training, not strictly cross-dataset

def parse_sizes_md(file_path: str = None) -> Dict[str, DatasetSizes]:
    """Parse sizes.md file and return dataset information"""
    if file_path is None:
        # Default to sizes.md in parent directory if called from tools/
        script_dir = pathlib.Path(__file__).parent
        file_path = script_dir.parent / "sizes.md"
    
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"sizes.md not found at {file_path}")
    
    content = path.read_text()
    
    # Parse overview table
    overview_pattern = r"\| \*\*([^*]+)\*\* \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \| ([\d.]+) \| ([\d.]+) \|"
    overview_matches = re.findall(overview_pattern, content)
    
    # Parse test set statistics
    test_pattern = r"\| \*\*([^*]+)\*\* \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \| ([\d.]+)% \|"
    test_section = content.split("## Test Set Statistics")[1].split("## Validation Set Statistics")[0]
    test_matches = re.findall(test_pattern, test_section)
    
    datasets = {}
    
    # Combine overview and test data
    overview_dict = {name: data for name, *data in overview_matches}
    test_dict = {name: data for name, *data in test_matches}
    
    for dataset_name in overview_dict:
        if dataset_name in test_dict:
            overview_data = overview_dict[dataset_name]
            test_data = test_dict[dataset_name]
            
            datasets[dataset_name] = DatasetSizes(
                dataset=dataset_name,
                table_a_records=int(overview_data[0].replace(',', '')),
                table_b_records=int(overview_data[1].replace(',', '')),
                total_records=int(overview_data[2].replace(',', '')),
                avg_tokens_a=float(overview_data[3]),
                avg_tokens_b=float(overview_data[4]),
                test_pairs=int(test_data[0].replace(',', '')),
                positive_pairs=int(test_data[1].replace(',', '')),
                negative_pairs=int(test_data[2].replace(',', '')),
                positive_rate=float(test_data[3])
            )
    
    return datasets

def parse_leaderboard_md(file_path: str = None) -> Dict[str, List[LeaderboardEntry]]:
    """Parse leaderboard.md file and return per-dataset leaderboards"""
    if file_path is None:
        # Default to leaderboard.md in parent directory if called from tools/
        script_dir = pathlib.Path(__file__).parent
        file_path = script_dir.parent / "leaderboard.md"
    
    path = pathlib.Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"leaderboard.md not found at {file_path}")
    
    content = path.read_text()
    
    # Split into dataset sections
    sections = re.split(r'### ([a-z_\\]+)', content)[1:]  # Skip first empty element
    
    leaderboards = {}
    
    for i in range(0, len(sections), 2):
        dataset_name = sections[i].strip()
        table_content = sections[i + 1]
        
        # Clean up dataset name (remove markdown escapes)
        clean_name = dataset_name.replace('\\_', '_')
        
        # Parse table rows
        row_pattern = r'\| (\d+) \| ([^|]+) \| ([^|]+) \|'
        rows = re.findall(row_pattern, table_content)
        
        entries = []
        for rank, model, f1_str in rows:
            # Clean up model name and F1 score
            model = model.strip()
            f1_str = f1_str.strip()
            
            # Check for special formatting
            is_best = f1_str.startswith('**') and f1_str.endswith('**')
            is_jellyfish = model.startswith('*') and model.endswith('*')
            
            # Extract F1 score
            f1_clean = re.sub(r'[*]', '', f1_str)
            f1_value = float(f1_clean)
            
            # Clean model name
            model_clean = re.sub(r'[*]', '', model)
            
            entries.append(LeaderboardEntry(
                rank=int(rank),
                model=model_clean,
                f1=f1_value,
                is_best=is_best,
                is_jellyfish=is_jellyfish
            ))
        
        leaderboards[clean_name] = entries
    
    return leaderboards

def estimate_max_safe_candidates(dataset_name: str, max_tokens: int = 1_000_000) -> Tuple[int, float]:
    """
    Estimate maximum safe candidate ratio to avoid token limit errors.
    
    Returns:
        (max_candidates, max_ratio): Maximum absolute candidates and ratio
    """
    sizes = parse_sizes_md()
    
    if dataset_name not in sizes:
        raise ValueError(f"Dataset {dataset_name} not found in sizes.md")
    
    dataset = sizes[dataset_name]
    
    # Estimate tokens per record (using average of A and B)
    avg_tokens_per_record = (dataset.avg_tokens_a + dataset.avg_tokens_b) / 2
    
    # Rough estimate: prompt overhead + left record + candidates
    # Left record: ~avg_tokens_per_record
    # Each candidate: ~avg_tokens_per_record + JSON formatting overhead (~20 tokens)
    # Prompt text: ~500 tokens
    prompt_overhead = 500
    left_record_tokens = avg_tokens_per_record
    tokens_per_candidate = avg_tokens_per_record + 20
    
    available_tokens = max_tokens - prompt_overhead - left_record_tokens
    max_candidates = int(available_tokens / tokens_per_candidate)
    
    # Cap at table B size
    max_candidates = min(max_candidates, dataset.table_b_records)
    
    # Calculate ratio
    max_ratio = max_candidates / dataset.table_b_records
    
    return max_candidates, max_ratio

def get_competitive_f1_threshold(dataset_name: str, percentile: float = 0.8) -> float:
    """
    Get F1 threshold to be considered competitive on a dataset.
    
    Args:
        dataset_name: Name of the dataset
        percentile: What percentile of the leaderboard to target (0.8 = top 20%)
    
    Returns:
        F1 threshold to be competitive
    """
    leaderboards = parse_leaderboard_md()
    
    # Map common abbreviations to full names
    name_mapping = {
        'abt_buy': 'abt',
        'amazon_google': 'amazon_google', 
        'dblp_acm': 'dblp_acm',
        'dblp_scholar': 'dblp_scholar', 
        'fodors_zagat': 'fodors_zagat',
        'zomato_yelp': 'zomato_yelp',
        'itunes_amazon': 'itunes_amazon',
        'rotten_imdb': 'rotten_imdb',
        'walmart_amazon': 'walmart_amazon'
    }
    
    # Try direct name first, then mapping
    lookup_name = dataset_name
    if dataset_name in name_mapping:
        lookup_name = name_mapping[dataset_name]
    
    if lookup_name not in leaderboards:
        # Try reverse mapping
        for full_name, abbrev in name_mapping.items():
            if abbrev == dataset_name and full_name in leaderboards:
                lookup_name = full_name
                break
    
    if lookup_name not in leaderboards:
        raise ValueError(f"Dataset {dataset_name} not found in leaderboard.md")
    
    entries = leaderboards[lookup_name]
    
    # Filter out jellyfish results (not strictly cross-dataset)
    cross_dataset_entries = [e for e in entries if not e.is_jellyfish]
    
    if not cross_dataset_entries:
        # Fallback to all entries if no cross-dataset ones
        cross_dataset_entries = entries
    
    f1_scores = [e.f1 for e in cross_dataset_entries]
    f1_scores.sort(reverse=True)
    
    # Get the percentile threshold
    index = int(len(f1_scores) * (1 - percentile))
    index = max(0, min(index, len(f1_scores) - 1))
    
    return f1_scores[index]

if __name__ == "__main__":
    # Test the parsing
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Show info for specific dataset")
    parser.add_argument("--test-parsing", action="store_true", help="Test parsing both files")
    args = parser.parse_args()
    
    if args.test_parsing:
        print("Testing sizes.md parsing...")
        sizes = parse_sizes_md()
        print(f"Parsed {len(sizes)} datasets")
        
        print("\nTesting leaderboard.md parsing...")
        leaderboards = parse_leaderboard_md()
        print(f"Parsed {len(leaderboards)} dataset leaderboards")
        
        print("\nDataset overview:")
        for name in sorted(sizes.keys()):
            size_info = sizes[name]
            print(f"{name}: {size_info.table_b_records:,} records B, {size_info.test_pairs:,} test pairs")
    
    if args.dataset:
        sizes = parse_sizes_md()
        leaderboards = parse_leaderboard_md()
        
        if args.dataset in sizes:
            info = sizes[args.dataset]
            print(f"\n=== {args.dataset.upper()} ===")
            print(f"Table B records: {info.table_b_records:,}")
            print(f"Test pairs: {info.test_pairs:,}")
            print(f"Avg tokens/record: {info.avg_tokens_b:.1f}")
            
            max_candidates, max_ratio = estimate_max_safe_candidates(args.dataset)
            print(f"Max safe candidates: {max_candidates:,} ({max_ratio:.1%})")
            
            try:
                threshold = get_competitive_f1_threshold(args.dataset)
                print(f"Competitive F1 threshold (80th percentile): {threshold:.1f}")
            except ValueError as e:
                print(f"Leaderboard lookup failed: {e}")
        else:
            print(f"Dataset {args.dataset} not found")