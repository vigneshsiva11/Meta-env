"""
server/environment.py
=====================
API Contract Negotiator — Core Environment Logic

Three real-world tasks (easy → medium → hard), deterministic consumer
test simulation, partial reward signals at every step, and all seven
unique features baked in.

Unique features implemented here
---------------------------------
1. Structured episode logging hooks (called from inference.py)
2. Seed-based reproducibility on reset()
3. Consumer simulation with per-field hard-break detection
4. Step-level hint system (hints off → proves pure RL learnability)
5. SUPPORTS_CONCURRENT_SESSIONS = True (4 parallel sessions)
6. Deprecation-header bonus reward (Task-02)
7. Force-termination on catastrophic failure (≥5 hard breaks)
"""
from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import ContractAction, ContractObservation, ContractState


# ============================================================================
# Helpers
# ============================================================================

def _f(
    name: str,
    ftype: str,
    required: bool = True,
    deprecated: bool = False,
    alias: Optional[str] = None,
    description: str = "",
) -> Dict[str, Any]:
    """Build a field dict for the schema."""
    return {
        "name": name,
        "type": ftype,
        "required": required,
        "deprecated": deprecated,
        "alias": alias,
        "description": description,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task catalogue
# ──────────────────────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {

    # ── TASK-01 ─ Easy ────────────────────────────────────────────────────────
    # Add a missing "currency" field to the payments API.
    # Two consumers already in production must not break.
    "TASK-01": {
        "difficulty": "easy",
        "description": (
            "The payments API response is missing a 'currency' field. "
            "Add it as an optional string field (type: str, required: false) "
            "without breaking the two existing consumers: "
            "'dashboard_app' and 'mobile_app'."
        ),
        "max_steps": 6,
        "initial_schema": [
            _f("transaction_id", "str",   description="Unique TX UUID"),
            _f("amount",         "float", description="Amount charged"),
            _f("status",         "str",   description="pending|success|failed"),
            _f("created_at",     "str",   description="ISO-8601 timestamp"),
        ],
        "consumers": {
            "dashboard_app": {
                "reads":       ["transaction_id", "amount", "status", "created_at"],
                "expects_new": [],
            },
            "mobile_app": {
                "reads":       ["transaction_id", "amount", "status"],
                "expects_new": [],
            },
        },
        "forward_requirements": [
            {"field": "currency", "present": True, "required": False},
        ],
        "bonus_deprecation_header": False,
    },

    # ── TASK-02 ─ Medium ──────────────────────────────────────────────────────
    # Rename "user_name" → "username" (snake_case unification).
    # Three legacy consumers read the OLD name — agent must add a
    # backward-compat alias. Bonus reward for emitting deprecation header.
    "TASK-02": {
        "difficulty": "medium",
        "description": (
            "The user profile API uses 'user_name'. Rename it to 'username'. "
            "Three legacy consumers ('legacy_portal', 'analytics_service', "
            "'mobile_client') still read 'user_name' — keep it working as an "
            "alias. One new consumer ('new_dashboard') expects 'username'. "
            "Setting add_deprecation_header=true earns a +0.05 bonus."
        ),
        "max_steps": 10,
        "initial_schema": [
            _f("user_id",    "str", description="Unique user UUID"),
            _f("user_name",  "str", description="Display name (legacy)"),
            _f("email",      "str", description="Email address"),
            _f("created_at", "str", description="Account creation ISO-8601"),
            _f("role",       "str", description="admin|user|guest"),
        ],
        "consumers": {
            "legacy_portal": {
                "reads":       ["user_id", "user_name", "email"],
                "expects_new": [],
            },
            "analytics_service": {
                "reads":       ["user_id", "user_name", "role"],
                "expects_new": [],
            },
            "mobile_client": {
                "reads":       ["user_id", "user_name", "created_at"],
                "expects_new": [],
            },
            "new_dashboard": {
                "reads":       [],
                "expects_new": [{"field": "username", "present": True}],
            },
        },
        "forward_requirements": [
            {"field": "username",  "present": True,  "required": True},
            {"field": "user_name", "present": True,  "required": False},
        ],
        "bonus_deprecation_header": True,
    },

    # ── TASK-03 ─ Hard ────────────────────────────────────────────────────────
    # Merge two conflicting API versions (v1 Team-A vs v2 Team-B)
    # into one unified schema that satisfies all four consumers.
    "TASK-03": {
        "difficulty": "hard",
        "description": (
            "Two client teams use different versions of the order API. "
            "Team A (legacy) reads: order_id, customer_id, total_price, shipped. "
            "Team B (modern) expects: id, buyer_id, total_amount, is_shipped, "
            "tracking_number. "
            "Unify both into one schema — add aliases, new fields, or both — "
            "without removing any field a consumer currently reads. "
            "All four consumers must reach 100% test pass-rate."
        ),
        "max_steps": 15,
        "initial_schema": [
            _f("order_id",    "str",   description="Order UUID (v1)"),
            _f("customer_id", "str",   description="Customer UUID (v1)"),
            _f("total_price", "float", description="Total in USD (v1)"),
            _f("shipped",     "bool",  description="Shipment flag (v1)"),
        ],
        "consumers": {
            "team_a_checkout": {
                "reads":       ["order_id", "customer_id", "total_price", "shipped"],
                "expects_new": [],
            },
            "team_a_reporting": {
                "reads":       ["order_id", "total_price"],
                "expects_new": [],
            },
            "team_b_fulfilment": {
                "reads":       [],
                "expects_new": [
                    {"field": "id",             "present": True},
                    {"field": "buyer_id",        "present": True},
                    {"field": "total_amount",    "present": True},
                    {"field": "is_shipped",      "present": True},
                    {"field": "tracking_number", "present": True},
                ],
            },
            "team_b_notifications": {
                "reads":       [],
                "expects_new": [
                    {"field": "id",        "present": True},
                    {"field": "buyer_id",  "present": True},
                    {"field": "is_shipped","present": True},
                ],
            },
        },
        "forward_requirements": [
            {"field": "id",             "present": True},
            {"field": "buyer_id",       "present": True},
            {"field": "total_amount",   "present": True},
            {"field": "is_shipped",     "present": True},
            {"field": "tracking_number","present": True},
        ],
        "bonus_deprecation_header": False,
    },
}

TASK_ORDER: List[str] = ["TASK-01", "TASK-02", "TASK-03"]


# ──────────────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────────────

class ApiContractEnvironment(Environment):
    """
    API Contract Negotiator — OpenEnv Environment.

    The agent evolves a REST API response schema through typed mutations
    (add_field, rename_field, add_alias, …) to satisfy new requirements
    while keeping existing consumer test suites at 100%.

    Grading is fully deterministic — no LLM required inside the environment.
    """

    # Unique feature #5: concurrent session support
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Unique feature #7: catastrophic-failure threshold
    HARD_BREAK_TERMINATION_THRESHOLD: int = 5
    REWARD_MIN: float = 0.0001
    REWARD_MAX: float = 0.9999

    def __init__(self) -> None:
        super().__init__()
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._task_id: str = ""
        self._task: Dict[str, Any] = {}
        self._schema: List[Dict[str, Any]] = []
        self._mutations: int = 0
        self._hard_breaks_total: int = 0
        self._submitted: bool = False
        self._deprecation_header_active: bool = False

    # ── reset() ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ContractObservation:
        """
        Start a new episode.

        Parameters
        ----------
        seed    : int | None  — for reproducibility (unique feature #2)
        task_id : str | None  — "TASK-01" | "TASK-02" | "TASK-03"
                                defaults to "TASK-01"
        """
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._mutations = 0
        self._hard_breaks_total = 0
        self._submitted = False
        self._deprecation_header_active = False

        self._task_id = task_id if task_id in TASKS else "TASK-01"
        self._task = TASKS[self._task_id]
        self._schema = copy.deepcopy(self._task["initial_schema"])

        consumer_results = self._run_consumer_tests()
        bw, fw, nr = self._compute_scores(consumer_results)

        return ContractObservation(
            done=False,
            reward=self._clamp_open_reward(bw * 0.30),
            current_schema=copy.deepcopy(self._schema),
            action_applied="Episode started — initial schema loaded.",
            action_valid=True,
            consumer_results=consumer_results,
            backward_compat_score=self._clamp_open_score(bw),
            forward_compat_score=self._clamp_open_score(fw),
            no_redundancy_score=self._clamp_open_score(nr),
            deprecation_header_present=self._deprecation_header_active,
            task_id=self._task_id,
            task_difficulty=self._task["difficulty"],
            steps_remaining=self._task["max_steps"],
            hint=self._task["description"],          # full task description on reset
        )

    # ── step() ───────────────────────────────────────────────────────────────

    def step(
        self,
        action: ContractAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ContractObservation:
        """Apply one schema mutation and return a scored observation."""
        self._step_count += 1
        max_steps: int = self._task["max_steps"]
        steps_remaining = max(0, max_steps - self._step_count)

        # Apply the mutation
        valid, msg, new_schema = self._apply_mutation(action)

        if not valid:
            consumer_results = self._run_consumer_tests()
            bw, fw, nr = self._compute_scores(consumer_results)
            done = steps_remaining == 0
            reward = self._clamp_open_reward(max(0.0, bw * 0.30 - 0.05))
            return ContractObservation(
                done=done,
                reward=reward,
                current_schema=copy.deepcopy(self._schema),
                action_applied=msg,
                action_valid=False,
                consumer_results=consumer_results,
                backward_compat_score=self._clamp_open_score(bw),
                forward_compat_score=self._clamp_open_score(fw),
                no_redundancy_score=self._clamp_open_score(nr),
                deprecation_header_present=self._deprecation_header_active,
                task_id=self._task_id,
                task_difficulty=self._task["difficulty"],
                steps_remaining=steps_remaining,
                hint="Invalid action — schema unchanged.",
            )

        self._schema = new_schema
        self._mutations += 1

        consumer_results = self._run_consumer_tests()
        hard_breaks_this_step = sum(r["hard_breaks"] for r in consumer_results)
        self._hard_breaks_total += hard_breaks_this_step

        bw, fw, nr = self._compute_scores(consumer_results)

        is_submit = action.action_type == "submit"
        force_end = (
            steps_remaining == 0
            or self._hard_breaks_total >= self.HARD_BREAK_TERMINATION_THRESHOLD
        )
        done = is_submit or force_end

        if done:
            self._submitted = is_submit
            reward = self._final_reward(bw, fw, nr)
        else:
            reward = self._clamp_open_reward(bw * 0.30)

        hint = "" if done else self._build_hint(bw, fw, nr, hard_breaks_this_step)

        return ContractObservation(
            done=done,
            reward=reward,
            current_schema=copy.deepcopy(self._schema),
            action_applied=msg,
            action_valid=True,
            consumer_results=consumer_results,
            backward_compat_score=self._clamp_open_score(bw),
            forward_compat_score=self._clamp_open_score(fw),
            no_redundancy_score=self._clamp_open_score(nr),
            deprecation_header_present=self._deprecation_header_active,
            task_id=self._task_id,
            task_difficulty=self._task["difficulty"],
            steps_remaining=steps_remaining,
            hint=hint,
        )

    # ── state property ───────────────────────────────────────────────────────

    @property
    def state(self) -> ContractState:
        return ContractState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            task_difficulty=self._task.get("difficulty", "easy"),
            max_steps=self._task.get("max_steps", 10),
            mutations_applied=self._mutations,
            hard_breaks_total=self._hard_breaks_total,
            submitted=self._submitted,
            deprecation_header_active=self._deprecation_header_active,
        )

    # ── Mutation engine ───────────────────────────────────────────────────────

    def _apply_mutation(
        self, action: ContractAction
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Validate and apply an action to the schema.
        Returns (valid, message, new_schema).
        """
        schema = copy.deepcopy(self._schema)
        at = action.action_type
        tf = action.target_field

        def find(n: str) -> Optional[Dict[str, Any]]:
            for f in schema:
                if f["name"] == n or f.get("alias") == n:
                    return f
            return None

        if at == "submit":
            return True, "Agent submitted the final contract.", schema

        if at == "add_field":
            if find(tf):
                return False, f"Field '{tf}' already exists.", schema
            schema.append(_f(
                tf,
                action.new_type or "str",
                required=False,
                description=action.reasoning[:120],
            ))
            return True, f"Added optional field '{tf}' (type: {action.new_type or 'str'}).", schema

        if at == "rename_field":
            if not action.new_name:
                return False, "rename_field requires new_name.", schema
            target = find(tf)
            if not target:
                return False, f"Field '{tf}' not found.", schema
            if find(action.new_name):
                return False, f"Field '{action.new_name}' already exists.", schema
            old_name = target["name"]
            target["name"] = action.new_name
            target["alias"] = old_name        # auto backward-compat alias
            if action.add_deprecation_header:
                self._deprecation_header_active = True
            dep_note = " Deprecation header activated." if action.add_deprecation_header else ""
            return True, f"Renamed '{old_name}' → '{action.new_name}' (alias kept: '{old_name}').{dep_note}", schema

        if at == "remove_field":
            if not find(tf):
                return False, f"Field '{tf}' not found.", schema
            schema = [f for f in schema if f["name"] != tf]
            return True, f"Removed field '{tf}'. WARNING: potential hard breaks.", schema

        if at == "add_alias":
            if not action.new_name:
                return False, "add_alias requires new_name.", schema
            target = find(tf)
            if not target:
                return False, f"Field '{tf}' not found.", schema
            schema.append(_f(
                action.new_name,
                target.get("type", "str"),
                required=False,
                alias=tf,
                description=f"Alias for '{tf}'",
            ))
            return True, f"Added alias '{action.new_name}' → '{tf}'.", schema

        if at == "change_type":
            if not action.new_type:
                return False, "change_type requires new_type.", schema
            target = find(tf)
            if not target:
                return False, f"Field '{tf}' not found.", schema
            old_t = target.get("type", "str")
            target["type"] = action.new_type
            return True, f"Changed type of '{tf}': {old_t} → {action.new_type}.", schema

        if at == "mark_deprecated":
            target = find(tf)
            if not target:
                return False, f"Field '{tf}' not found.", schema
            target["deprecated"] = True
            if action.add_deprecation_header:
                self._deprecation_header_active = True
            dep_note = " Deprecation header activated." if action.add_deprecation_header else ""
            return True, f"Marked '{tf}' as deprecated.{dep_note}", schema

        return False, f"Unknown action_type '{at}'.", schema

    # ── Consumer test simulator ───────────────────────────────────────────────

    def _run_consumer_tests(self) -> List[Dict[str, Any]]:
        """
        Unique feature #3 — per-consumer, per-field hard-break detection.

        A field is "accessible" if its name OR its alias appears in the schema.
        A hard break = a consumer READ field that has disappeared with no alias.
        """
        schema_names  = {f["name"] for f in self._schema}
        schema_aliases = {f.get("alias") for f in self._schema if f.get("alias")}
        accessible = schema_names | schema_aliases

        results = []
        consumers: Dict[str, Dict] = self._task.get("consumers", {})

        for consumer_name, spec in consumers.items():
            reads: List[str] = spec.get("reads", [])
            expects_new: List[Dict] = spec.get("expects_new", [])

            tests_total = len(reads) + len(expects_new)
            tests_passed = 0
            hard_breaks = 0

            for field_name in reads:
                if field_name in accessible:
                    tests_passed += 1
                else:
                    hard_breaks += 1   # field the consumer reads is gone

            for exp in expects_new:
                fname = exp["field"]
                should_present = exp.get("present", True)
                if (fname in accessible) == should_present:
                    tests_passed += 1

            score = tests_passed / tests_total if tests_total > 0 else 1.0
            results.append({
                "consumer":     consumer_name,
                "tests_passed": tests_passed,
                "tests_total":  tests_total,
                "hard_breaks":  hard_breaks,
                "score":        self._clamp_open_score(score),
            })

        return results

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_scores(
        self, consumer_results: List[Dict[str, Any]]
    ) -> Tuple[float, float, float]:
        """Returns (backward_compat, forward_compat, no_redundancy)."""
        consumers = self._task.get("consumers", {})

        # backward: average score of consumers that READ old fields
        bw_scores = [
            r["score"]
            for r in consumer_results
            if consumers.get(r["consumer"], {}).get("reads")
        ]
        backward = sum(bw_scores) / len(bw_scores) if bw_scores else 1.0

        # forward: how many forward_requirements are satisfied
        fwd_reqs = self._task.get("forward_requirements", [])
        accessible = (
            {f["name"] for f in self._schema}
            | {f.get("alias") for f in self._schema if f.get("alias")}
        )
        if fwd_reqs:
            met = sum(
                1 for req in fwd_reqs
                if req.get("present", True) == (req["field"] in accessible)
            )
            forward = met / len(fwd_reqs)
        else:
            forward = 1.0

        # redundancy penalty: excessive mutations beyond expected minimum
        expected = max(1, len(fwd_reqs))
        excess = max(0, self._mutations - expected * 2)
        redundancy = max(0.0, 1.0 - excess * 0.10)

        return (
            self._clamp_open_score(backward),
            self._clamp_open_score(forward),
            self._clamp_open_score(redundancy),
        )

    def _final_reward(self, bw: float, fw: float, nr: float) -> float:
        """
        Final composite reward.

        reward = bw*0.50 + fw*0.30 + nr*0.20
               − 0.10 per hard break (floor 0.0)
               + 0.05 bonus if deprecation header present (Task-02 only)
        """
        base = bw * 0.50 + fw * 0.30 + nr * 0.20
        penalty = self._hard_breaks_total * 0.10
        reward = max(self.REWARD_MIN, base - penalty)
        if self._task.get("bonus_deprecation_header") and self._deprecation_header_active:
            reward = min(self.REWARD_MAX, reward + 0.05)
        return self._clamp_open_reward(reward)

    def _clamp_open_reward(self, reward: float) -> float:
        """Ensure rewards stay in the strict open interval required by validator."""
        return round(min(self.REWARD_MAX, max(self.REWARD_MIN, reward)), 4)

    def _clamp_open_score(self, score: float) -> float:
        """Ensure score-like metrics stay in the strict open interval required by validator."""
        return round(min(self.REWARD_MAX, max(self.REWARD_MIN, score)), 4)

    def _build_hint(
        self, bw: float, fw: float, nr: float, hard_breaks: int
    ) -> str:
        """Unique feature #4 — step-level contextual hint."""
        parts: List[str] = []
        if bw < 0.5:
            parts.append("Backward compat is low — add aliases before renaming/removing fields.")
        if fw < 0.5:
            parts.append("Forward requirements not yet met — review the task description.")
        if hard_breaks > 0:
            parts.append(f"{hard_breaks} hard break(s) this step — consumers lost field access.")
        if nr < 0.7:
            parts.append("Too many mutations — try to minimise unnecessary changes.")
        if not parts:
            parts.append("Looking good — consider submitting when both scores reach 0.9999.")
        return " | ".join(parts)
