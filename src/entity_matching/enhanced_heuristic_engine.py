#!/usr/bin/env python
"""
Enhanced heuristic rule engine with sophisticated control logic.

This module supports:
1. Score rules - Adjust similarity scores
2. Pipeline rules - Control pipeline flow and weights
3. Decision rules - Make early termination decisions
4. Weight rules - Dynamic weight adjustment based on conditions
"""

import json
import os
import pathlib
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum
import importlib.util
import sys


class RuleType(Enum):
    SCORE = "score"
    PIPELINE = "pipeline" 
    DECISION = "decision"
    WEIGHT = "weight"


class PipelineStage(Enum):
    CANDIDATE_SELECTION = "candidate_selection"
    PRE_SEMANTIC = "pre_semantic"
    POST_SEMANTIC = "post_semantic"
    PRE_LLM = "pre_llm"
    POST_LLM = "post_llm"


@dataclass
class ScoreAction:
    """Action to adjust similarity scores"""
    score_adjustment: float
    action_type: str = "score_adjustment"
    confidence: float = 1.0
    reason: str = ""


@dataclass
class WeightAction:
    """Action to adjust matching weights"""
    action_type: str = "weight_adjustment"
    confidence: float = 1.0
    reason: str = ""
    semantic_weight: Optional[float] = None
    trigram_weight: Optional[float] = None


@dataclass
class DecisionAction:
    """Action to make early termination decisions"""
    terminate_early: bool
    action_type: str = "early_decision"
    confidence: float = 1.0
    reason: str = ""
    final_result: Optional[int] = None  # 1 = match, 0 = no match, None = continue
    skip_llm: bool = False


@dataclass
class PipelineAction:
    """Action to control pipeline flow"""
    action_type: str = "pipeline_control"
    confidence: float = 1.0
    reason: str = ""
    skip_stage: bool = False
    modify_candidates: bool = False
    candidate_override: Optional[List] = None


@dataclass
class EnhancedRule:
    """Enhanced rule with sophisticated control capabilities"""
    rule_name: str
    rule_type: RuleType
    description: str
    implementation: str
    confidence: float
    stage: PipelineStage
    test_cases: List[Dict[str, Any]]
    compiled_function: Optional[Callable] = None


class EnhancedHeuristicEngine:
    """Enhanced heuristic engine with sophisticated control logic"""
    
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.score_rules: List[EnhancedRule] = []
        self.pipeline_rules: List[EnhancedRule] = []
        self.decision_rules: List[EnhancedRule] = []
        self.weight_rules: List[EnhancedRule] = []
        self.compiled_rules: Dict[str, Callable] = {}
        
    def load_enhanced_heuristics(self, heuristic_file: str) -> int:
        """Load enhanced heuristic rules from JSON file"""
        if not os.path.exists(heuristic_file):
            print(f"⚠️  Enhanced heuristic file not found: {heuristic_file}")
            return 0
        
        try:
            with open(heuristic_file, 'r') as f:
                data = json.load(f)
            
            total_loaded = 0
            
            # Load different rule types
            for rule_type_name, rules_list in [
                ("score_rules", data.get('score_rules', [])),
                ("pipeline_rules", data.get('pipeline_rules', [])), 
                ("decision_rules", data.get('decision_rules', [])),
                ("weight_rules", data.get('weight_rules', []))
            ]:
                rule_type = RuleType(rule_type_name.replace('_rules', ''))
                
                for rule_data in rules_list:
                    rule = EnhancedRule(
                        rule_name=rule_data['rule_name'],
                        rule_type=rule_type,
                        description=rule_data['description'],
                        implementation=rule_data['implementation'],
                        confidence=rule_data['confidence'],
                        stage=PipelineStage(rule_data.get('stage', 'candidate_selection')),
                        test_cases=rule_data.get('test_cases', [])
                    )
                    
                    # Compile the rule function
                    if self._compile_enhanced_rule(rule):
                        if rule_type == RuleType.SCORE:
                            self.score_rules.append(rule)
                        elif rule_type == RuleType.PIPELINE:
                            self.pipeline_rules.append(rule)
                        elif rule_type == RuleType.DECISION:
                            self.decision_rules.append(rule)
                        elif rule_type == RuleType.WEIGHT:
                            self.weight_rules.append(rule)
                        
                        total_loaded += 1
                        print(f"  ✅ {rule.rule_name} ({rule_type.value}, {rule.stage.value})")
                    else:
                        print(f"  ❌ Failed to compile {rule.rule_name}")
            
            print(f"📋 Loaded {total_loaded} enhanced rules: "
                  f"{len(self.score_rules)} score, {len(self.pipeline_rules)} pipeline, "
                  f"{len(self.decision_rules)} decision, {len(self.weight_rules)} weight")
            
            return total_loaded
            
        except Exception as e:
            print(f"Error loading enhanced heuristics: {e}")
            return 0
    
    def _compile_enhanced_rule(self, rule: EnhancedRule) -> bool:
        """Compile an enhanced heuristic rule function"""
        try:
            # Create a temporary module to execute the function
            module_name = f"enhanced_heuristic_{rule.rule_name}"
            
            # Add necessary imports and action classes to the function code
            function_code = f"""
import re
import math
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

def normalize(text):
    \"\"\"Normalize text by removing punctuation, converting to lowercase, and stripping whitespace\"\"\"
    if not text:
        return ''
    # Remove punctuation, convert to lowercase, and strip whitespace
    text = re.sub(r'[^\\w\\s]', '', str(text).lower().strip())
    return ' '.join(text.split())

# Import action classes
class RuleAction:
    def __init__(self, action_type, confidence=1.0, reason=""):
        self.action_type = action_type
        self.confidence = confidence
        self.reason = reason

class ScoreAction(RuleAction):
    def __init__(self, score_adjustment, confidence=1.0, reason=""):
        super().__init__("score_adjustment", confidence, reason)
        self.score_adjustment = score_adjustment

class WeightAction(RuleAction):
    def __init__(self, semantic_weight=None, trigram_weight=None, confidence=1.0, reason=""):
        super().__init__("weight_adjustment", confidence, reason)
        self.semantic_weight = semantic_weight
        self.trigram_weight = trigram_weight

class DecisionAction(RuleAction):
    def __init__(self, terminate_early, final_result=None, skip_llm=False, confidence=1.0, reason=""):
        super().__init__("early_decision", confidence, reason)
        self.terminate_early = terminate_early
        self.final_result = final_result
        self.skip_llm = skip_llm

class PipelineAction(RuleAction):
    def __init__(self, skip_stage=False, modify_candidates=False, candidate_override=None, confidence=1.0, reason=""):
        super().__init__("pipeline_control", confidence, reason)
        self.skip_stage = skip_stage
        self.modify_candidates = modify_candidates
        self.candidate_override = candidate_override

{rule.implementation}
"""
            
            # Compile and execute the function
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            
            exec(function_code, module.__dict__)
            
            # Get the compiled function
            function_name = rule.rule_name
            if hasattr(module, function_name):
                rule.compiled_function = getattr(module, function_name)
                self.compiled_rules[rule.rule_name] = rule.compiled_function
                return True
            else:
                print(f"    Warning: Function {function_name} not found in compiled code")
                return False
                
        except Exception as e:
            print(f"    Error compiling enhanced rule {rule.rule_name}: {e}")
            return False
    
    def apply_score_rules(self, left_record: Dict[str, Any], right_record: Dict[str, Any], 
                         stage: PipelineStage) -> float:
        """Apply score adjustment rules"""
        total_adjustment = 0.0
        
        for rule in self.score_rules:
            if rule.stage == stage and rule.compiled_function:
                try:
                    result = rule.compiled_function(left_record, right_record)
                    if isinstance(result, (int, float)):
                        # Legacy score adjustment
                        weighted_adjustment = result * rule.confidence
                        total_adjustment += weighted_adjustment
                    elif hasattr(result, 'score_adjustment'):
                        # ScoreAction from compiled rule (duck typing for class compatibility)
                        weighted_adjustment = result.score_adjustment * rule.confidence
                        total_adjustment += weighted_adjustment
                except Exception as e:
                    print(f"    Warning: Score rule {rule.rule_name} failed: {e}")
                    continue
        
        return total_adjustment
    
    def apply_decision_rules(self, left_record: Dict[str, Any], right_record: Dict[str, Any],
                           current_score: float, stage: PipelineStage) -> Optional[DecisionAction]:
        """Apply decision rules for early termination"""
        for rule in self.decision_rules:
            if rule.stage == stage and rule.compiled_function:
                try:
                    result = rule.compiled_function(left_record, right_record, current_score)
                    # Check for DecisionAction duck typing (class name might differ due to compilation)
                    if hasattr(result, 'terminate_early') and hasattr(result, 'final_result'):
                        # It's a DecisionAction from the compiled rule - return it directly
                        return result
                    elif isinstance(result, DecisionAction):
                        return result
                    elif result is not None:
                        # Legacy boolean result
                        return DecisionAction(
                            terminate_early=bool(result),
                            confidence=rule.confidence,
                            reason=f"Legacy rule {rule.rule_name}"
                        )
                except Exception as e:
                    print(f"    Warning: Decision rule {rule.rule_name} failed: {e}")
                    continue
        
        return None
    
    def apply_weight_rules(self, left_record: Dict[str, Any], right_record: Dict[str, Any],
                          current_weights: Dict[str, float], stage: PipelineStage) -> Optional[WeightAction]:
        """Apply weight adjustment rules"""
        for rule in self.weight_rules:
            if rule.stage == stage and rule.compiled_function:
                try:
                    result = rule.compiled_function(left_record, right_record, current_weights)
                    # Check for WeightAction duck typing (class name might differ due to compilation)
                    if hasattr(result, 'semantic_weight') or hasattr(result, 'trigram_weight'):
                        # It's a WeightAction from compiled rule - return it directly
                        return result
                    elif isinstance(result, WeightAction):
                        return result
                except Exception as e:
                    print(f"    Warning: Weight rule {rule.rule_name} failed: {e}")
                    continue
        
        return None
    
    def apply_pipeline_rules(self, left_record: Dict[str, Any], right_record: Dict[str, Any],
                           stage: PipelineStage) -> Optional[PipelineAction]:
        """Apply pipeline control rules"""
        for rule in self.pipeline_rules:
            if rule.stage == stage and rule.compiled_function:
                try:
                    result = rule.compiled_function(left_record, right_record)
                    if isinstance(result, PipelineAction):
                        return result
                except Exception as e:
                    print(f"    Warning: Pipeline rule {rule.rule_name} failed: {e}")
                    continue
        
        return None
    
    def get_enhanced_rule_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get information about loaded enhanced rules"""
        return {
            'score_rules': [
                {
                    'rule_name': rule.rule_name,
                    'description': rule.description,
                    'confidence': rule.confidence,
                    'stage': rule.stage.value,
                    'compiled': rule.compiled_function is not None
                }
                for rule in self.score_rules
            ],
            'pipeline_rules': [
                {
                    'rule_name': rule.rule_name,
                    'description': rule.description,
                    'confidence': rule.confidence,
                    'stage': rule.stage.value,
                    'compiled': rule.compiled_function is not None
                }
                for rule in self.pipeline_rules
            ],
            'decision_rules': [
                {
                    'rule_name': rule.rule_name,
                    'description': rule.description,
                    'confidence': rule.confidence,
                    'stage': rule.stage.value,
                    'compiled': rule.compiled_function is not None
                }
                for rule in self.decision_rules
            ],
            'weight_rules': [
                {
                    'rule_name': rule.rule_name,
                    'description': rule.description,
                    'confidence': rule.confidence,
                    'stage': rule.stage.value,
                    'compiled': rule.compiled_function is not None
                }
                for rule in self.weight_rules
            ]
        }


def load_enhanced_heuristics_for_dataset(dataset: str, heuristic_file: Optional[str] = None) -> EnhancedHeuristicEngine:
    """Load enhanced heuristics for a specific dataset"""
    engine = EnhancedHeuristicEngine(dataset)
    
    if heuristic_file:
        # Use specified file
        engine.load_enhanced_heuristics(heuristic_file)
    else:
        # Try to find enhanced heuristics file for this dataset
        possible_files = [
            f"results/generated_rules/{dataset}_generated_heuristics.json",
            f"{dataset}_enhanced_heuristics.json",
            f"enhanced_heuristics_{dataset}.json",
            f"data/enhanced_heuristics/{dataset}.json"
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                engine.load_enhanced_heuristics(file_path)
                break
        else:
            print(f"⚠️  No enhanced heuristics file found for dataset '{dataset}'")
            print(f"    Tried: {possible_files}")
    
    return engine


def test_enhanced_heuristic_engine():
    """Test the enhanced heuristic engine"""
    print("🧪 TESTING ENHANCED HEURISTIC ENGINE")
    
    # Create sample enhanced heuristics
    sample_heuristics = {
        "score_rules": [
            {
                "rule_name": "brewery_exact_match_boost",
                "description": "Strong boost when brewery names match exactly",
                "implementation": """
def brewery_exact_match_boost(left_record, right_record):
    left_brewery = left_record.get('Brew_Factory_Name', '').strip().lower()
    right_brewery = right_record.get('Brew_Factory_Name', '').strip().lower()
    
    if left_brewery and right_brewery and left_brewery == right_brewery:
        return ScoreAction(score_adjustment=0.5, confidence=0.95, reason="Exact brewery match")
    return ScoreAction(score_adjustment=0.0)
""",
                "confidence": 0.95,
                "stage": "candidate_selection",
                "test_cases": []
            }
        ],
        "decision_rules": [
            {
                "rule_name": "high_confidence_early_decision",
                "description": "Skip LLM for very high or very low similarity scores",
                "implementation": """
def high_confidence_early_decision(left_record, right_record, current_score):
    if current_score > 0.95:
        return DecisionAction(
            terminate_early=True,
            final_result=1,
            confidence=0.9,
            reason="Very high similarity score"
        )
    elif current_score < 0.1:
        return DecisionAction(
            terminate_early=True,
            final_result=0,
            skip_llm=True,
            confidence=0.8,
            reason="Very low similarity, skip expensive LLM"
        )
    return None
""",
                "confidence": 0.85,
                "stage": "pre_llm",
                "test_cases": []
            }
        ],
        "weight_rules": [
            {
                "rule_name": "style_mismatch_weight_adjustment",
                "description": "Increase semantic weight when beer styles are incompatible",
                "implementation": """
def style_mismatch_weight_adjustment(left_record, right_record, current_weights):
    left_style = left_record.get('Style', '').lower()
    right_style = right_record.get('Style', '').lower()
    
    # Check for incompatible styles
    incompatible_pairs = [
        (['lager', 'pilsner'], ['stout', 'porter']),
        (['wheat', 'weizen'], ['ipa'])
    ]
    
    for group1, group2 in incompatible_pairs:
        left_in_g1 = any(term in left_style for term in group1)
        right_in_g2 = any(term in right_style for term in group2)
        left_in_g2 = any(term in left_style for term in group2)
        right_in_g1 = any(term in right_style for term in group1)
        
        if (left_in_g1 and right_in_g2) or (left_in_g2 and right_in_g1):
            return WeightAction(
                semantic_weight=0.9,
                confidence=0.8,
                reason="Compensating for style incompatibility with higher semantic weight"
            )
    
    return None
""",
                "confidence": 0.75,
                "stage": "pre_semantic",
                "test_cases": []
            }
        ],
        "pipeline_rules": []
    }
    
    # Save sample heuristics
    with open("test_enhanced_heuristics.json", "w") as f:
        json.dump(sample_heuristics, f, indent=2)
    
    # Test loading
    engine = load_enhanced_heuristics_for_dataset("test", "test_enhanced_heuristics.json")
    
    # Test with sample data
    sample_left = {
        "Beer_Name": "Amber Ale",
        "Brew_Factory_Name": "Mountain Goat Beer",
        "ABV": "5.0%",
        "Style": "American Amber Ale"
    }
    
    sample_right = {
        "Beer_Name": "Red Ale", 
        "Brew_Factory_Name": "Mountain Goat Beer",
        "ABV": "5.2%",
        "Style": "Lager"
    }
    
    print(f"\n🧪 TESTING WITH SAMPLE DATA:")
    print(f"Left: {sample_left}")
    print(f"Right: {sample_right}")
    
    # Test score rules
    score_adj = engine.apply_score_rules(sample_left, sample_right, PipelineStage.CANDIDATE_SELECTION)
    print(f"Score adjustment: {score_adj:.3f}")
    
    # Test weight rules
    current_weights = {"semantic_weight": 0.5}
    weight_action = engine.apply_weight_rules(sample_left, sample_right, current_weights, PipelineStage.PRE_SEMANTIC)
    if weight_action:
        print(f"Weight adjustment: semantic_weight -> {weight_action.semantic_weight}")
    
    # Test decision rules
    decision = engine.apply_decision_rules(sample_left, sample_right, 0.97, PipelineStage.PRE_LLM)
    if decision:
        print(f"Decision: terminate_early={decision.terminate_early}, result={decision.final_result}")
    
    # Show rule info
    print(f"\n📋 ENHANCED RULES SUMMARY:")
    rule_info = engine.get_enhanced_rule_info()
    for rule_type, rules in rule_info.items():
        print(f"  {rule_type}: {len(rules)} rules")
        for rule in rules:
            print(f"    - {rule['rule_name']} ({rule['stage']})")
    
    # Cleanup
    os.remove("test_enhanced_heuristics.json")
    
    return engine


if __name__ == "__main__":
    test_enhanced_heuristic_engine()