#!/usr/bin/env python3
"""Quick score bounds test - minimal version"""
import sys
sys.path.insert(0, '.')

from server.environment import ApiContractEnvironment
from models import ContractAction

print("Testing score bounds...")

env = ApiContractEnvironment()

for task_id in ["TASK-01", "TASK-02", "TASK-03"]:
    print(f"\n{task_id}:")
    env.reset(task_id=task_id, seed=42)
    action = ContractAction(action_type="submit", target_field="_")
    obs = env.step(action)
    score = obs.reward
    
    in_bounds = 0.0 < score < 1.0
    status = "✅ PASS" if in_bounds else "❌ FAIL"
    print(f"  Score: {score:.4f} {status}")
    
    if not in_bounds:
        print(f"    ERROR: Score must be strictly between 0 and 1, not {score}")
        sys.exit(1)

print("\n✅ ALL TESTS PASSED - All scores are strictly inside (0, 1)")
