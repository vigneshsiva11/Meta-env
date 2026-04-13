#!/usr/bin/env python3
"""
Direct test of inference.py score clamping functions
"""
import sys
import os

# Set dummy HF_TOKEN to allow import
os.environ['HF_TOKEN'] = 'dummy-key-for-testing'

sys.path.insert(0, '.')

from inference import (
    REWARD_MIN, REWARD_MAX, _clamp_open_reward, _display_open_score
)

print("=" * 70)
print("Testing inference.py score clamping functions")
print("=" * 70)

# Test cases: (input_score, description)
test_cases = [
    (0.0, "Exactly 0.0 (boundary)"),
    (0.00001, "Very small value"),
    (0.0001, "REWARD_MIN (0.0001)"),
    (0.5, "Middle value"),
    (0.7000, "Typical good score"),
    (0.8499, "Typical good score"),
    (0.9999, "REWARD_MAX (0.9999)"),
    (1.0, "Exactly 1.0 (boundary)"),
    (1.1, "Above 1.0"),
    (-0.1, "Negative value"),
]

print("\nTesting _clamp_open_reward():")
print(f"  REWARD_MIN = {REWARD_MIN}")
print(f"  REWARD_MAX = {REWARD_MAX}")
print()

all_pass = True
for input_val, description in test_cases:
    clamped = _clamp_open_reward(input_val)
    displayed = _display_open_score(input_val)
    
    is_valid = REWARD_MIN <= clamped <= REWARD_MAX and 0.0 < clamped < 1.0
    status = "✅" if is_valid else "❌"
    
    print(f"{status} Input: {input_val:7.4f} → Clamped: {clamped:.4f}, Displayed: {displayed:.4f}  ({description})")
    
    if not is_valid:
        print(f"    ERROR: Output {clamped} violates open interval (0, 1)")
        all_pass = False

print()
print("=" * 70)

# Test the actual environment scores
print("\nTesting environment scores:")
from server.environment import ApiContractEnvironment
from models import ContractAction

env = ApiContractEnvironment()
env_scores = []

for task_id in ["TASK-01", "TASK-02", "TASK-03"]:
    env.reset(task_id=task_id, seed=42)
    action = ContractAction(action_type="submit", target_field="_")
    obs = env.step(action)
    
    raw_score = obs.reward
    clamped_score = _clamp_open_reward(raw_score)
    displayed = _display_open_score(raw_score)
    
    is_valid = 0.0 < clamped_score < 1.0
    status = "✅" if is_valid else "❌"
    
    print(f"{status} {task_id}: raw={raw_score:.4f} → clamped={clamped_score:.4f} → display={displayed:.4f}")
    env_scores.append(clamped_score)
    
    if not is_valid:
        all_pass = False

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

if all_pass:
    print("✅ SUCCESS - All scores are strictly inside (0, 1)")
    print("\nInference.py is ready for deployment!")
    sys.exit(0)
else:
    print("❌ FAILURE - Some scores violated bounds")
    sys.exit(1)
