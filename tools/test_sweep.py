#!/usr/bin/env python
"""
Quick test of the sweeping functionality in mock mode.
"""

import subprocess
import os

def test_mock_sweep():
    """Test sweep functionality without using real API calls"""
    
    # Ensure we're in mock mode
    env = os.environ.copy()
    if 'OPENAI_API_KEY' in env:
        del env['OPENAI_API_KEY']
    
    print("Testing sweep in MOCK mode (no API calls)...")
    
    cmd = [
        "python", "tools/sweep_candidates.py",
        "--dataset", "beer",
        "--limit", "10",
        "--num-points", "3",
        "--output", "test_sweep_results.json"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, env=env, timeout=120)  # 2 min timeout
        
        if result.returncode == 0:
            print("✅ Mock sweep completed successfully!")
            
            # Check if results file was created
            import pathlib
            if pathlib.Path("test_sweep_results.json").exists():
                print("✅ Results file created")
                
                # Show results
                import json
                with open("test_sweep_results.json") as f:
                    results = json.load(f)
                
                print(f"✅ Generated {len(results)} results")
                if results:
                    best = max(results, key=lambda x: x.get('f1', 0))
                    print(f"✅ Best F1: {best.get('f1', 0):.1f} at {best['candidate_ratio']:.1%} ratio")
            else:
                print("❌ Results file not created")
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print("❌ Command timed out")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_mock_sweep()