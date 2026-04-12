"""
client.py
=========
Typed client for the API Contract Negotiator environment.

Usage (sync)
------------
    from client import ApiContractEnv
    from models import ContractAction

    env = ApiContractEnv(base_url="http://localhost:8000").sync()
    with env:
        result = env.reset(task_id="TASK-01")
        while not result.done:
            action = ContractAction(action_type="add_field", target_field="currency")
            result = env.step(action)
        print(f"Final reward: {result.reward}")

Usage (async)
-------------
    async with ApiContractEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="TASK-02")
        result = await env.step(action)
"""
from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import ContractAction, ContractObservation, ContractState


class ApiContractEnv(EnvClient[ContractAction, ContractObservation, ContractState]):
    """Typed synchronous/async client for ApiContractEnvironment."""

    def _step_payload(self, action: ContractAction) -> Dict[str, Any]:
        """Serialise action → dict for WebSocket transport."""
        return action.model_dump(exclude_none=False)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ContractObservation]:
        """Deserialise WebSocket response → StepResult[ContractObservation]."""
        obs_data: Dict[str, Any] = payload.get("observation", payload)
        obs = ContractObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=float(payload.get("reward", obs.reward or 0.0)),
            done=bool(payload.get("done", obs.done)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ContractState:
        """Deserialise state response → ContractState."""
        return ContractState(
            episode_id=payload.get("episode_id"),
            step_count=int(payload.get("step_count", 0)),
            task_id=str(payload.get("task_id", "")),
            task_difficulty=payload.get("task_difficulty", "easy"),
            max_steps=int(payload.get("max_steps", 10)),
            mutations_applied=int(payload.get("mutations_applied", 0)),
            hard_breaks_total=int(payload.get("hard_breaks_total", 0)),
            submitted=bool(payload.get("submitted", False)),
            deprecation_header_active=bool(payload.get("deprecation_header_active", False)),
        )
