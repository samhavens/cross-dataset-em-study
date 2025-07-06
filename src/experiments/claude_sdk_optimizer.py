#!/usr/bin/env python
"""
Claude Code SDK integration for intelligent hyperparameter optimization.
"""

import json
import os
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass 
class OptimizationSuggestion:
    """A suggested hyperparameter configuration from Claude analysis"""
    max_candidates: int
    semantic_weight: float
    model: str
    use_semantic: bool
    reasoning: str
    priority: int  # 1=high, 2=medium, 3=low
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "semantic_weight": self.semantic_weight,
            "model": self.model,
            "use_semantic": self.use_semantic,
            "reasoning": self.reasoning,
            "priority": self.priority
        }


class ClaudeSDKOptimizer:
    """Uses Claude Code SDK to analyze results and suggest next hyperparameters"""
    
    def __init__(self):
        self.claude_executable = self._find_claude_executable()
        
    def _find_claude_executable(self) -> Optional[str]:
        """Find the Claude Code SDK executable"""
        # Try common locations
        possible_paths = [
            "claude",  # If in PATH
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            "~/.local/bin/claude"
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path) or self._command_exists(path):
                return path
                
        print("Warning: Claude Code SDK not found. Install with: pip install claude-sdk")
        return None
    
    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in PATH"""
        try:
            subprocess.run([command, "--version"], 
                         capture_output=True, 
                         timeout=5)
            return True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def analyze_and_suggest(self, analysis_text: str) -> List[OptimizationSuggestion]:
        """Use Claude Code SDK to analyze results and suggest next configurations"""
        if not self.claude_executable:
            print("Claude Code SDK not available, falling back to heuristic suggestions")
            return self._fallback_suggestions(analysis_text)
        
        # Create prompt for Claude
        prompt = self._create_optimization_prompt(analysis_text)
        
        try:
            # Call Claude Code SDK
            result = self._call_claude_sdk(prompt)
            return self._parse_claude_response(result)
            
        except Exception as e:
            print(f"Error calling Claude Code SDK: {e}")
            return self._fallback_suggestions(analysis_text)
    
    def _create_optimization_prompt(self, analysis_text: str) -> str:
        """Create a detailed prompt for Claude to analyze hyperparameter results"""
        prompt = f"""You are an expert at hyperparameter optimization for entity matching systems. 

I've run an initial hyperparameter sweep for an entity matching system that uses:
1. Trigram similarity for fast candidate filtering
2. Optional semantic similarity (sentence transformers) for better quality
3. LLM (GPT) for final binary matching decisions

The goal is to achieve F1 > 0.912 (91.2%) which is the competitive threshold.

Here are the results from my initial sweep:

{analysis_text}

Based on these results, please suggest 5-8 new hyperparameter configurations to test next. 
Consider:
1. Performance patterns you observe
2. Cost-effectiveness 
3. Areas of the hyperparameter space not yet explored
4. Configurations that might push us over the 91.2% threshold

For each suggestion, provide:
- max_candidates: integer (1-200)
- semantic_weight: float (0.0-1.0, where 0.0=trigram only, 1.0=semantic only)  
- model: "gpt-4.1-nano" or "gpt-4.1-mini"
- use_semantic: boolean
- reasoning: why this configuration might work better
- priority: 1=high priority, 2=medium, 3=exploratory

Please respond in this exact JSON format:
{{
  "suggestions": [
    {{
      "max_candidates": 20,
      "semantic_weight": 0.6,
      "model": "gpt-4.1-nano", 
      "use_semantic": true,
      "reasoning": "Your analysis here",
      "priority": 1
    }}
  ]
}}
"""
        return prompt
    
    def _call_claude_sdk(self, prompt: str) -> str:
        """Call Claude Code SDK with the given prompt"""
        try:
            # Call Claude Code SDK with --print flag for non-interactive mode
            result = subprocess.run(
                [self.claude_executable, "--print", prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Claude SDK error: {result.stderr}")
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude SDK call timed out")
        except Exception as e:
            raise RuntimeError(f"Claude SDK call failed: {e}")
    
    def _parse_claude_response(self, response: str) -> List[OptimizationSuggestion]:
        """Parse Claude's JSON response into OptimizationSuggestion objects"""
        try:
            # Extract JSON from response (might have other text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            data = json.loads(json_str)
            suggestions = []
            
            for item in data.get("suggestions", []):
                suggestion = OptimizationSuggestion(
                    max_candidates=item["max_candidates"],
                    semantic_weight=item["semantic_weight"],
                    model=item["model"],
                    use_semantic=item["use_semantic"],
                    reasoning=item["reasoning"],
                    priority=item["priority"]
                )
                suggestions.append(suggestion)
            
            return suggestions
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing Claude response: {e}")
            print(f"Raw response: {response}")
            return self._fallback_suggestions("")
    
    def _fallback_suggestions(self, analysis_text: str) -> List[OptimizationSuggestion]:
        """Fallback heuristic suggestions if Claude SDK is not available"""
        print("Using fallback heuristic suggestions...")
        
        suggestions = [
            OptimizationSuggestion(
                max_candidates=15,
                semantic_weight=0.6,
                model="gpt-4.1-nano",
                use_semantic=True,
                reasoning="Balanced approach with slightly more semantic weight",
                priority=1
            ),
            OptimizationSuggestion(
                max_candidates=30,
                semantic_weight=0.4,
                model="gpt-4.1-nano", 
                use_semantic=True,
                reasoning="More candidates with moderate semantic weight",
                priority=1
            ),
            OptimizationSuggestion(
                max_candidates=10,
                semantic_weight=0.8,
                model="gpt-4.1-mini",
                use_semantic=True,
                reasoning="High semantic weight with better model",
                priority=2
            ),
            OptimizationSuggestion(
                max_candidates=75,
                semantic_weight=0.3,
                model="gpt-4.1-nano",
                use_semantic=True,
                reasoning="Many candidates, low semantic weight for speed",
                priority=2
            ),
            OptimizationSuggestion(
                max_candidates=5,
                semantic_weight=0.9,
                model="gpt-4.1-mini",
                use_semantic=True,
                reasoning="Extreme semantic focus with premium model",
                priority=3
            )
        ]
        
        return suggestions


def test_claude_sdk():
    """Test Claude SDK integration with sample data"""
    optimizer = ClaudeSDKOptimizer()
    
    sample_analysis = """
    # Test Analysis
    Dataset: beer
    Total configurations tested: 3
    
    ## Top 3 Performing Configurations:
    1. F1=0.8500, Model=gpt-4.1-nano, Candidates=10, Semantic_weight=0.5, Cost=$0.0100
    2. F1=0.8200, Model=gpt-4.1-nano, Candidates=25, Semantic_weight=0.0, Cost=$0.0150  
    3. F1=0.8000, Model=gpt-4.1-mini, Candidates=10, Semantic_weight=0.7, Cost=$0.0300
    """
    
    suggestions = optimizer.analyze_and_suggest(sample_analysis)
    
    print("Optimization Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. Priority {suggestion.priority}: {suggestion.model}, "
              f"{suggestion.max_candidates} candidates, semantic_weight={suggestion.semantic_weight}")
        print(f"   Reasoning: {suggestion.reasoning}")
        print()


if __name__ == "__main__":
    test_claude_sdk()