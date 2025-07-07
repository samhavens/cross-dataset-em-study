#!/usr/bin/env python3
"""
Agentic Claude Code SDK heuristic generation system.

Clean, agentic approach where Claude can:
1. Generate rules
2. Test them on sample data  
3. See results
4. Iterate and improve
5. Validate final performance

NO file swapping, NO subprocess calls to claude binary.
Uses only claude_code_sdk Python imports.
"""

import asyncio
import json
import os
import pathlib
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from ..entity_matching.hybrid_matcher import run_matching


def get_leaderboard_target_f1(dataset: str) -> float:
    """Get the top F1 score from leaderboard.md for the given dataset"""
    try:
        # Dataset name mappings (dataset -> leaderboard section name)
        dataset_mappings = {
            'abt_buy': 'abt',
            'dblp_acm': 'dblp\\_acm',
            'dblp_scholar': 'dblp\\_scholar', 
            'fodors_zagat': 'fodors\\_zagat',
            'zomato_yelp': 'zomato\\_yelp',
            'amazon_google': 'amazon\\_google',
            'beer': 'beer',
            'itunes_amazon': 'itunes\\_amazon',
            'rotten_imdb': 'rotten\\_imdb',
            'walmart_amazon': 'walmart\\_amazon'
        }
        
        # Read the leaderboard file
        leaderboard_path = pathlib.Path('leaderboard.md')
        if not leaderboard_path.exists():
            print(f"⚠️ leaderboard.md not found, using default target of 85.0")
            return 85.0
            
        with open(leaderboard_path, 'r') as f:
            content = f.read()
        
        # Get the dataset section name
        section_name = dataset_mappings.get(dataset, dataset)
        
        # Find the dataset section and extract highest F1 score
        import re
        section_pattern = rf"### {re.escape(section_name)}(?:\s+\([^)]+\))?\s*\n"
        match = re.search(section_pattern, content, re.IGNORECASE)
        
        if not match:
            print(f"⚠️ Dataset {dataset} not found in leaderboard, using default target of 85.0")
            return 85.0
        
        # Extract the section content until the next ### or end of file
        start = match.end()
        next_section = re.search(r"\n### ", content[start:])
        if next_section:
            section_content = content[start:start + next_section.start()]
        else:
            section_content = content[start:]
        
        # Find the highest F1 score in bold (non-italicized)
        f1_scores = []
        for line in section_content.split('\n'):
            if '|' in line and any(char.isdigit() for char in line):
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 3:  # model | F1 at minimum
                    f1_text = parts[-1]  # Last column is F1
                    
                    # Skip italicized scores (jellyfish)
                    if f1_text.startswith('*') and f1_text.endswith('*'):
                        continue
                    
                    # Extract number from **92.4** or 92.4
                    f1_match = re.search(r'(\d+\.?\d*)', f1_text)
                    if f1_match:
                        f1_scores.append(float(f1_match.group(1)))
        
        if f1_scores:
            target_f1 = max(f1_scores)
            print(f"🎯 Leaderboard target for {dataset}: F1 = {target_f1:.1f}")
            return target_f1
        else:
            print(f"⚠️ No F1 scores found for {dataset}, using default target of 85.0")
            return 85.0
            
    except Exception as e:
        print(f"⚠️ Error parsing leaderboard for {dataset}: {e}, using default target of 85.0")
        return 85.0


@dataclass
class SampleData:
    """Clean sample data for Claude to analyze and test rules on"""
    dataset: str
    dev_pairs: List[Dict[str, Any]]  # Sampled dev pairs with left/right records
    dev_predictions: Dict[int, int]  # Model predictions on dev set
    dev_metrics: Dict[str, float]  # F1, precision, recall on dev set
    target_f1: float  # Leaderboard target
    table_a: Dict[int, Dict]  # ID -> record mapping
    table_b: Dict[int, Dict]  # ID -> record mapping


class AgenticHeuristicGenerator:
    """Clean agentic heuristic generator using only claude_code_sdk"""
    
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.data_root = pathlib.Path('data') / 'raw' / dataset
        self.target_f1 = get_leaderboard_target_f1(dataset)
        
    def _load_clean_data(self) -> Tuple[Dict[int, Dict], Dict[int, Dict], pd.DataFrame]:
        """Load dataset cleanly without file manipulation"""
        # Load tables
        A_df = pd.read_csv(self.data_root / 'tableA.csv')
        B_df = pd.read_csv(self.data_root / 'tableB.csv')
        
        # Create ID-to-record mappings
        A = {row['id']: row.to_dict() for _, row in A_df.iterrows()}
        B = {row['id']: row.to_dict() for _, row in B_df.iterrows()}
        
        # Load dev set (validation if available, otherwise train slice)
        if (self.data_root / 'valid.csv').exists():
            dev_pairs = pd.read_csv(self.data_root / 'valid.csv')
            print(f"✅ Using validation set: {len(dev_pairs)} pairs")
        elif (self.data_root / 'train.csv').exists():
            train_pairs = pd.read_csv(self.data_root / 'train.csv')
            dev_slice_size = min(100, len(train_pairs))
            dev_pairs = train_pairs.head(dev_slice_size)
            print(f"✅ Using train slice: {dev_slice_size} pairs")
        else:
            raise ValueError("No validation or training set available for clean rule generation")
        
        return A, B, dev_pairs
    
    def _create_sample_data(self, dev_results: Dict[str, Any]) -> SampleData:
        """Create clean sample data for Claude to work with"""
        A, B, dev_pairs = self._load_clean_data()
        
        # Extract predictions and metrics - NEVER use mock data
        predictions = dev_results.get('predictions', {})
        metrics = dev_results.get('metrics', {})
        
        if not predictions:
            raise ValueError(f"No predictions found in dev_results. Cannot generate rules without real model predictions. Got keys: {list(dev_results.keys())}")
        
        print(f"📊 Real model predictions: {len(predictions)} predictions")
        
        # Create enriched sample pairs with actual records
        sample_pairs = []
        for _, row in dev_pairs.head(50).iterrows():  # Sample first 50 for analysis
            left_id = row.ltable_id
            right_id = row.rtable_id
            true_label = row.label
            
            # Get REAL model prediction - no fallbacks or mocks
            if left_id in predictions:
                predicted_right_id = predictions[left_id]
                predicted_match = (predicted_right_id == right_id)
                predicted_label = 1 if predicted_match else 0
            else:
                # If no prediction for this pair, skip it (model didn't evaluate this pair)
                continue
            
            # Categorize the prediction
            if true_label == 1 and predicted_label == 1:
                category = 'true_positive'
            elif true_label == 0 and predicted_label == 1:
                category = 'false_positive'  # Model error
            elif true_label == 1 and predicted_label == 0:
                category = 'false_negative'  # Model error
            else:
                category = 'true_negative'
            
            sample_pairs.append({
                'left_id': left_id,
                'right_id': right_id,
                'left_record': A[left_id],
                'right_record': B[right_id],
                'true_label': true_label,
                'predicted_label': predicted_label,
                'category': category
            })
        
        return SampleData(
            dataset=self.dataset,
            dev_pairs=sample_pairs,
            dev_predictions=predictions,
            dev_metrics=metrics,
            target_f1=self.target_f1,
            table_a=A,
            table_b=B
        )
    
    def _create_agentic_prompt(self, sample_data: SampleData) -> str:
        """Create agentic prompt that lets Claude test and iterate on rules"""
        
        # Analyze sample data for patterns
        error_examples = [p for p in sample_data.dev_pairs if p['category'] in ['false_positive', 'false_negative']]
        success_examples = [p for p in sample_data.dev_pairs if p['category'] in ['true_positive', 'true_negative']]
        
        prompt = f"""You are an expert at entity matching rule generation. You can iteratively develop and test rules.

**DATASET**: {sample_data.dataset}
**TARGET**: F1 > {sample_data.target_f1:.1f} (leaderboard target)
**CURRENT DEV PERFORMANCE**: F1={sample_data.dev_metrics.get('f1', 0):.4f}, P={sample_data.dev_metrics.get('precision', 0):.4f}, R={sample_data.dev_metrics.get('recall', 0):.4f}

**YOUR TOOLS**:
- `Read`: Read files (e.g., sample data, existing rules)
- `Write`: Write rules to test files
- `Bash`: Test rules by running matching with your generated rules

**TASK**: Generate sophisticated entity matching rules that improve F1 score through:
1. **Early decisions** (auto-accept/reject to reduce LLM costs)
2. **Score adjustments** (boost likely matches, penalize unlikely ones)  
3. **Dynamic weights** (adjust semantic vs trigram weights based on context)

**RULE TYPES**:
```python
# 1. DECISION RULES - Early termination to skip LLM
DecisionAction(terminate_early=True, final_result=1, confidence=0.95, reason="Exact match")
DecisionAction(terminate_early=True, final_result=0, skip_llm=True, reason="Very low similarity")

# 2. SCORE RULES - Adjust similarity scores  
ScoreAction(score_adjustment=0.3, confidence=0.8, reason="Strong field match")

# 3. WEIGHT RULES - Dynamic weight adjustment
WeightAction(semantic_weight=0.9, confidence=0.7, reason="Text-heavy comparison")
```

**PIPELINE STAGES**:
- `pre_llm`: Best for early decisions (auto-accept/reject)
- `post_semantic`: After semantic similarity, good for score adjustments
- `pre_semantic`: Before semantic calculation, good for weight adjustments

**SAMPLE DATA ANALYSIS**:
Model made {len(error_examples)} errors ({len([p for p in error_examples if p['category']=='false_positive'])} false positives, {len([p for p in error_examples if p['category']=='false_negative'])} false negatives)

**FALSE POSITIVE EXAMPLES** (incorrectly predicted as matches):
"""
        
        # Add false positive examples
        fp_examples = [p for p in error_examples if p['category'] == 'false_positive'][:3]
        for i, example in enumerate(fp_examples):
            prompt += f"""
Example {i+1}:
Left:  {json.dumps(example['left_record'], indent=2)}
Right: {json.dumps(example['right_record'], indent=2)}
"""
        
        prompt += f"""
**FALSE NEGATIVE EXAMPLES** (missed real matches):
"""
        
        # Add false negative examples  
        fn_examples = [p for p in error_examples if p['category'] == 'false_negative'][:3]
        for i, example in enumerate(fn_examples):
            prompt += f"""
Example {i+1}:
Left:  {json.dumps(example['left_record'], indent=2)}
Right: {json.dumps(example['right_record'], indent=2)}
"""

        prompt += f"""

**ITERATIVE WORKFLOW**:
1. **Write initial rules** to `test_rules.json` based on the error patterns above
2. **Test them** with: `python run_enhanced_matching.py --dataset {sample_data.dataset} --heuristic-file test_rules.json --limit 20`
3. **Analyze results** - did F1 improve? Which rules helped/hurt?
4. **Iterate** - refine rules and test again
5. **Final rules** - when satisfied, write final rules to `generated_rules.json`

**RULE FORMAT**:
```json
{{
  "score_rules": [
    {{
      "rule_name": "exact_name_boost",
      "description": "Boost score for exact name matches",
      "implementation": "def exact_name_boost(left_record, right_record):\\n    if normalize(left_record.get('name', '')) == normalize(right_record.get('name', '')):\\n        return ScoreAction(score_adjustment=0.4, confidence=0.9, reason='Exact name match')\\n    return None",
      "confidence": 0.9,
      "stage": "post_semantic"
    }}
  ],
  "decision_rules": [
    {{
      "rule_name": "very_low_similarity_reject", 
      "description": "Auto-reject very low similarity",
      "implementation": "def very_low_similarity_reject(left_record, right_record, current_score):\\n    if current_score < 0.1:\\n        return DecisionAction(terminate_early=True, final_result=0, skip_llm=True, confidence=0.95, reason='Very low similarity')\\n    return None",
      "confidence": 0.95,
      "stage": "pre_llm"
    }}
  ],
  "weight_rules": [],
  "pipeline_rules": []
}}
```

**START HERE**: Analyze the error patterns above, then write your first set of rules to `test_rules.json` and test them!
"""
        
        return prompt
    
    def _write_sample_data_file(self, sample_data: SampleData) -> str:
        """Write sample data to a file Claude can read"""
        sample_file = f"sample_data_{sample_data.dataset}.json"
        
        def clean_for_json(obj):
            """Convert pandas types to JSON-serializable types"""
            if hasattr(obj, 'item'):  # numpy/pandas scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            else:
                return obj
        
        # Prepare clean sample data
        clean_data = {
            'dataset': sample_data.dataset,
            'target_f1': sample_data.target_f1,
            'current_metrics': sample_data.dev_metrics,
            'sample_pairs': sample_data.dev_pairs[:20],  # First 20 for analysis
            'error_analysis': {
                'false_positives': [p for p in sample_data.dev_pairs if p['category'] == 'false_positive'][:5],
                'false_negatives': [p for p in sample_data.dev_pairs if p['category'] == 'false_negative'][:5],
                'true_positives': [p for p in sample_data.dev_pairs if p['category'] == 'true_positive'][:3],
                'true_negatives': [p for p in sample_data.dev_pairs if p['category'] == 'true_negative'][:3]
            }
        }
        
        # Clean all pandas types for JSON serialization
        clean_data = clean_for_json(clean_data)
        
        with open(sample_file, 'w') as f:
            json.dump(clean_data, f, indent=2)
        
        print(f"📊 Sample data written to {sample_file}")
        return sample_file
    
    async def _call_claude_agentic(self, prompt: str, sample_file: str) -> str:
        """Call Claude Code SDK in agentic mode with file access"""
        try:
            print(f"🤖 Starting agentic Claude session...")
            print(f"📊 Sample data: {sample_file}")
            
            # Add sample file info to prompt
            enhanced_prompt = f"""{prompt}

**SAMPLE DATA FILE**: `{sample_file}` contains detailed error analysis and examples.
You can read this file to see all the false positive/negative examples.

**TESTING COMMAND**: 
```bash
python run_enhanced_matching.py --dataset {self.dataset} --heuristic-file test_rules.json --limit 20 --max-candidates 50
```

Start by reading the sample data file to understand the patterns, then iteratively develop and test rules!
"""
            
            options = ClaudeCodeOptions(
                allowed_tools=["Read", "Write", "Bash"],
                permission_mode="acceptEdits",
                cwd=os.getcwd(),
            )
            
            response_parts = []
            async for message in query(prompt=enhanced_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    # end of run
                    break
            
            full_response = "".join(response_parts)
            print(f"✅ Claude session completed, response length: {len(full_response)} chars")
            return full_response
            
        except Exception as e:
            raise RuntimeError(f"Claude SDK agentic call failed: {e}")
    
    async def generate_agentic_rules(self, dev_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate rules using agentic Claude approach"""
        print(f"🚀 AGENTIC HEURISTIC GENERATION FOR {self.dataset}")
        print("=" * 60)
        
        # Prepare clean sample data
        sample_data = self._create_sample_data(dev_results)
        sample_file = self._write_sample_data_file(sample_data)
        
        # Create agentic prompt
        prompt = self._create_agentic_prompt(sample_data)
        
        try:
            # Run agentic Claude session
            response = await self._call_claude_agentic(prompt, sample_file)
            
            # Look for generated rules file
            possible_files = [
                'generated_rules.json',
                'test_rules.json', 
                f'{self.dataset}_generated_heuristics.json',
                'final_rules.json'
            ]
            
            generated_file = None
            for filename in possible_files:
                if os.path.exists(filename):
                    generated_file = filename
                    print(f"✅ Found generated rules: {filename}")
                    break
            
            if not generated_file:
                print("⚠️ No rules file found, Claude may have encountered issues")
                print(f"Response preview: {response[:500]}...")
                return None
            
            # Copy to final output location if specified
            if output_file and generated_file != output_file:
                with open(generated_file, 'r') as src:
                    rules_data = json.load(src)
                
                # Add metadata
                rules_data['timestamp'] = datetime.now().isoformat()
                rules_data['dataset'] = self.dataset
                rules_data['generation_method'] = 'agentic_claude_sdk'
                rules_data['target_f1'] = self.target_f1
                
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as dst:
                    json.dump(rules_data, dst, indent=2)
                
                print(f"✅ Final rules saved to: {output_file}")
                return output_file
            else:
                return generated_file
                
        except Exception as e:
            print(f"❌ Agentic rule generation failed: {e}")
            raise
        finally:
            # Clean up sample file
            if os.path.exists(sample_file):
                os.unlink(sample_file)


async def generate_agentic_heuristics(dataset: str, dev_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
    """
    Clean interface for agentic heuristic generation.
    
    Args:
        dataset: Dataset name
        dev_results: Development set results with predictions and metrics
        output_file: Optional output file path
        
    Returns:
        Path to generated heuristics file
    """
    generator = AgenticHeuristicGenerator(dataset)
    return await generator.generate_agentic_rules(dev_results, output_file)


if __name__ == "__main__":
    # Test CLI
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate agentic heuristics using Claude Code SDK")
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--dev-results', required=True, help='Path to dev results JSON file')
    parser.add_argument('--output', help='Output file for generated heuristics')
    
    args = parser.parse_args()
    
    # Load dev results
    with open(args.dev_results, 'r') as f:
        dev_results = json.load(f)
    
    async def main():
        output_file = await generate_agentic_heuristics(args.dataset, dev_results, args.output)
        print(f"🎉 Agentic heuristics generated: {output_file}")
    
    asyncio.run(main())