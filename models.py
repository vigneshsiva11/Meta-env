"""
models.py
=========
Typed Action / Observation / State for the API Contract Negotiator Env.

All classes inherit directly from the openenv.core.env_server.types base
classes so the framework's serialisation, validation, and WebSocket layer
work without any extra wiring.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


# ------------------------------------------------------------------------------
# Action
# ------------------------------------------------------------------------------

class ContractAction(Action):
    """
    One schema-mutation the agent wants to apply to the current API contract.

    action_type choices
    -------------------
    add_field        – add a new (optional) field to the response schema
    rename_field     – rename a field; must supply new_name; alias auto-added
    remove_field     – delete a field (risks hard consumer breaks)
    add_alias        – add an alias name that points to an existing field
    change_type      – change the Pydantic type of an existing field
    mark_deprecated  – soft-flag a field as deprecated
    submit           – finalise the contract and trigger full grading
    """

    action_type: Literal[
        "add_field",
        "rename_field",
        "remove_field",
        "add_alias",
        "change_type",
        "mark_deprecated",
        "submit",
    ] = Field(..., description="Schema mutation to perform")

    target_field: str = Field(
        ..., min_length=1, description="Field name the action operates on"
    )

    new_name: Optional[str] = Field(
        default=None, description="New name (rename_field / add_alias)"
    )

    new_type: Optional[str] = Field(
        default=None, description="New type string (change_type)"
    )

    add_deprecation_header: bool = Field(
        default=False,
        description="Emit X-Deprecated-Field header (bonus reward on Task-02)",
    )

    reasoning: str = Field(
        default="",
        max_length=500,
        description="Agent's explanation — logged, not scored",
    )


# ------------------------------------------------------------------------------
# Observation
# ------------------------------------------------------------------------------

class ContractObservation(Observation):
    """
    What the environment returns after every step() and reset().

    Inherits from openenv Observation:
      done   : bool
      reward : float | None
      metadata : dict
    """

    # Current schema state
    current_schema: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Live API response schema — list of field dicts",
    )

    # What happened this step
    action_applied: str = Field(
        default="", description="Human-readable description of the applied mutation"
    )
    action_valid: bool = Field(
        default=True, description="False if the action was structurally invalid"
    )

    # Consumer test results  (per-consumer breakdown)
    consumer_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-consumer test outcomes. Each entry: "
            "{'consumer': str, 'tests_passed': int, 'tests_total': int, "
            "'hard_breaks': int, 'score': float}"
        ),
    )

    # Partial scores (always present — enables intermediate learning signal)
    backward_compat_score: float = Field(
        default=0.0001, gt=0.0, lt=1.0,
        description="Fraction of old-consumer tests still passing",
    )
    forward_compat_score: float = Field(
        default=0.0001, gt=0.0, lt=1.0,
        description="Fraction of new requirements satisfied",
    )
    no_redundancy_score: float = Field(
        default=0.9999, gt=0.0, lt=1.0,
        description="1.0 = efficient; reduced by excessive mutations",
    )
    deprecation_header_present: bool = Field(
        default=False,
        description="Whether the active schema emits X-Deprecated-Field",
    )

    # Episode progress
    task_id: str = Field(default="", description="Current task scenario ID")
    task_difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    steps_remaining: int = Field(default=10, ge=0)
    hint: str = Field(
        default="",
        description="Guidance hint — empty when hint mode is off",
    )


# ------------------------------------------------------------------------------
# State
# ------------------------------------------------------------------------------

class ContractState(State):
    """
    Episode metadata returned by state().

    Inherits from openenv State:
      episode_id  : str | None
      step_count  : int
      extra='allow'
    """

    task_id: str = Field(default="")
    task_difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    max_steps: int = Field(default=10)
    mutations_applied: int = Field(default=0)
    hard_breaks_total: int = Field(default=0)
    submitted: bool = Field(default=False)
    deprecation_header_active: bool = Field(default=False)
