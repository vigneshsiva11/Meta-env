#!/usr/bin/env python3
"""
Test script to verify task scores are strictly inside (0, 1) boundary.
Does NOT require Gemini API key  tests grading logic directly.
"""
import sys
sys.path.insert(0, '.')

from server.environment import ApiContractEnvironment
from models import ContractAction


def test_open_interval_bounds():
    """Test that all task scores stay strictly inside (0, 1)."""
    print("=" * 70)
    print("TESTING STRICT OPEN INTERVAL (0, 1) BOUNDS")
    print("=" * 70)
    
    env = ApiContractEnvironment()
    
    test_cases = [
        ("Submit immediately (baseline)", [("submit", "_")]),
        ("One invalid action then submit", [("rename_field", "missing"), ("submit", "_")]),
        ("Multiple invalid actions then step limit", [
            ("rename_field", "missing"),
            ("rename_field", "missing"),
            ("rename_field", "missing"),
            ("rename_field", "missing"),
            ("rename_field", "missing"),
            ("rename_field", "missing"),
        ]),
    ]
    
    all_pass = True
    
    for task_id in ["TASK-01", "TASK-02", "TASK-03"]:
        print(f"\n{'' * 70}")
        print(f"Task: {task_id}")
        print(f"{'' * 70}")
        
        for test_name, actions in test_cases:
            env.reset(task_id=task_id, seed=42)
            
            print(f"\n  Test: {test_name}")
            
            obs = None
            for action_type, target_field in actions:
                action = ContractAction(
                    action_type=action_type,
                    target_field=target_field,
                )
                obs = env.step(action)
                
                if obs.done:
                    break
            
            if obs is not None:
                score = obs.reward
                print(f"    Final score: {score:.4f}")
                
                # Check bounds: must be strictly inside (0, 1)
                if score <= 0.0 or score >= 1.0:
                    print(f"     FAIL: Score {score:.4f} is NOT strictly inside (0, 1)")
                    all_pass = False
                elif not (0.0 < score < 1.0):
                    print(f"     FAIL: Score boundary check failed")
                    all_pass = False
                else:
                    print(f"     PASS: Score {score:.4f} is strictly inside (0, 1)")
            else:
                print(f"      WARNING: No observation returned")
    
    print(f"\n{'=' * 70}")
    if all_pass:
        print(" ALL TESTS PASSED - Task scores are safe for validator")
        return 0
    else:
        print(" SOME TESTS FAILED - Score bounds violated")
        return 1


def test_full_episode():
    """Run a complete episode on each task and verify final scores."""
    print("\n" + "=" * 70)
    print("TESTING COMPLETE EPISODES")
    print("=" * 70)
    
    env = ApiContractEnvironment()
    all_scores = []
    
    for task_id in ["TASK-01", "TASK-02", "TASK-03"]:
        print(f"\n{task_id}:")
        
        obs = env.reset(task_id=task_id, seed=42)
        step_count = 0
        
        # Run episode by always submitting on the first step for simplicity
        while not obs.done and step_count < 20:
            action = ContractAction(action_type="submit", target_field="_")
            obs = env.step(action)
            step_count += 1
        
        score = obs.reward
        all_scores.append(score)
        
        print(f"  Final reward: {score:.4f}")
        print(f"  Step count: {step_count}")
        
        # Validate
        if not (0.0 < score < 1.0):
            print(f"   FAIL: Score {score:.4f} violates (0, 1) bounds")
            return 1
        else:
            print(f"   PASS: Score is strictly inside (0, 1)")
    
    print(f"\n{'' * 70}")
    print(f"Summary:")
    for i, (task_id, score) in enumerate(zip(["TASK-01", "TASK-02", "TASK-03"], all_scores)):
        print(f"  {task_id}: {score:.4f}")
    avg = sum(all_scores) / len(all_scores)
    print(f"  Average: {avg:.4f}")
    
    if not (0.0 < avg < 1.0):
        print(f"   FAIL: Average {avg:.4f} violates (0, 1) bounds")
        return 1
    else:
        print(f"   PASS: Average is strictly inside (0, 1)")
    
    return 0


if __name__ == "__main__":
    result1 = test_open_interval_bounds()
    result2 = test_full_episode()
    
    print("\n" + "=" * 70)
    if result1 == 0 and result2 == 0:
        print(" ALL VALIDATION TESTS PASSED ")
        print("=" * 70)
        sys.exit(0)
    else:
        print(" SOME VALIDATION TESTS FAILED")
        print("=" * 70)
        sys.exit(1)
