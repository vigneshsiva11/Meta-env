"""
inference.py
============
Hackathon baseline inference script for the API Contract Negotiator Env.

Rules enforced
--------------
• Named inference.py at project root                        ✓
• Uses OpenAI client for all LLM calls                      ✓
• Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from env vars    ✓
• Emits [START] / [STEP] / [END] structured stdout logs     ✓
• Completes in < 20 min on 2 vCPU / 8 GB RAM               ✓
• Works with Gemini via the OpenAI-compatible endpoint      ✓

Gemini OpenAI-compatible endpoint
----------------------------------
  API_BASE_URL = https://generativelanguage.googleapis.com/v1beta/openai/
  MODEL_NAME   = gemini-2.0-flash
  HF_TOKEN     = <your Gemini API key from Google AI Studio>

Local test run
--------------
  export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
  export MODEL_NAME=gemini-2.0-flash
  export HF_TOKEN=<your-gemini-api-key>
  export ENV_BASE_URL=http://localhost:8000
  python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── local imports (project root is on sys.path) ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import ApiContractEnv
from models import ContractAction

# ── required env vars ─────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN:     str = os.environ["HF_TOKEN"]      # Gemini API key — no default, must be set
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# ── Gemini via OpenAI-compat client ──────────────────────────────────────────
#   Google's endpoint is fully compatible with openai.OpenAI.
#   Just set base_url to the Gemini OpenAI-compat URL and api_key to your
#   Gemini API key (stored in HF_TOKEN per hackathon spec).
llm = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,   # e.g. https://generativelanguage.googleapis.com/v1beta/openai/
)

TASKS_TO_RUN: List[str] = ["TASK-01", "TASK-02", "TASK-03"]

# ── Structured log helpers ────────────────────────────────────────────────────

def log_start(task_id: str, difficulty: str, max_steps: int) -> None:
    """Unique feature #1 — [START] log line."""
    print(
        f"[START] task_id={task_id} difficulty={difficulty} max_steps={max_steps}",
        flush=True,
    )


def log_step(
    step: int,
    action_type: str,
    target_field: str,
    reward: float,
    done: bool,
    backward: float,
    forward: float,
) -> None:
    """Unique feature #1 — [STEP] log line."""
    print(
        f"[STEP] step={step} action={action_type} target={target_field} "
        f"reward={reward:.4f} done={str(done).lower()} "
        f"backward_compat={backward:.4f} forward_compat={forward:.4f}",
        flush=True,
    )


def log_end(task_id: str, final_reward: float, total_steps: int) -> None:
    """Unique feature #1 — [END] log line."""
    print(
        f"[END] task_id={task_id} final_reward={final_reward:.4f} steps={total_steps}",
        flush=True,
    )


# ── LLM system prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior API engineer. Your job is to evolve a REST API response
schema to satisfy new requirements without breaking existing consumers.

You will receive the current schema, consumer test results, partial scores,
and the task description. Respond ONLY with a single JSON object:

{
  "action_type": "<add_field|rename_field|remove_field|add_alias|change_type|mark_deprecated|submit>",
  "target_field": "<field name to act on>",
  "new_name": "<only for rename_field or add_alias>",
  "new_type": "<only for change_type, e.g. str|int|float|bool|Optional[str]>",
  "add_deprecation_header": <true|false>,
  "reasoning": "<one sentence>"
}

Critical rules:
- NEVER remove a field a consumer reads without first adding an alias.
- When renaming a field any consumer depends on, ALWAYS set add_deprecation_header: true.
- When backward_compat_score == 1.0 AND forward_compat_score == 1.0, issue action_type "submit".
- Do NOT output markdown fences or any text outside the JSON object.
"""


def call_llm(obs_text: str, task_description: str) -> Optional[Dict[str, Any]]:
    """
    Call Gemini via the OpenAI-compatible endpoint and parse the action JSON.
    Returns None on any failure so the caller can fall back gracefully.
    """
    user_msg = f"Task:\n{task_description}\n\nCurrent environment state:\n{obs_text}"
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=400,
            temperature=0.0,
        )
        raw: str = response.choices[0].message.content.strip()

        # Strip markdown fences if Gemini wraps the JSON
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except json.JSONDecodeError as exc:
        print(f"  [WARN] JSON parse error: {exc} | raw={raw[:120]}", flush=True)
        return None
    except Exception as exc:
        print(f"  [WARN] LLM call failed: {exc}", flush=True)
        return None


def obs_to_text(obs: Any, step: int) -> str:
    """Format the observation into a concise LLM-readable string."""
    lines = [
        f"Step {step} | steps_remaining={obs.steps_remaining}",
        f"backward_compat={obs.backward_compat_score:.2f}  "
        f"forward_compat={obs.forward_compat_score:.2f}  "
        f"no_redundancy={obs.no_redundancy_score:.2f}",
        "",
        "Current schema:",
    ]
    for field in obs.current_schema:
        alias_note = f"  (alias: {field['alias']})" if field.get("alias") else ""
        dep_note   = "  [DEPRECATED]" if field.get("deprecated") else ""
        lines.append(
            f"  - {field['name']} : {field.get('type', 'str')}{alias_note}{dep_note}"
        )
    lines.append("")
    lines.append("Consumer test results:")
    for cr in obs.consumer_results:
        brk = f"  !! {cr['hard_breaks']} HARD BREAK(S)" if cr["hard_breaks"] else ""
        lines.append(
            f"  [{cr['consumer']}] "
            f"{cr['tests_passed']}/{cr['tests_total']} passed  "
            f"score={cr['score']:.2f}{brk}"
        )
    if obs.hint:
        lines.append(f"\nHint: {obs.hint}")
    return "\n".join(lines)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str, env: ApiContractEnv) -> float:
    """Run one task episode. Returns the final reward."""
    result = env.reset(task_id=task_id, seed=42)   # seed for reproducibility
    obs = result.observation

    task_description = obs.hint
    difficulty = obs.task_difficulty
    max_steps  = obs.steps_remaining

    log_start(task_id, difficulty, max_steps)

    step = 0
    final_reward: float = float(result.reward or 0.0)

    while not result.done:
        step += 1
        obs_text = obs_to_text(obs, step)
        action_data = call_llm(obs_text, task_description)

        # Fallback if LLM call failed
        if action_data is None:
            action_data = {
                "action_type": "submit",
                "target_field": "_",
                "reasoning": "LLM unavailable — submitting current state",
            }

        # Build and validate the typed action
        try:
            action = ContractAction(
                action_type=action_data.get("action_type", "submit"),
                target_field=action_data.get("target_field", "_"),
                new_name=action_data.get("new_name") or None,
                new_type=action_data.get("new_type") or None,
                add_deprecation_header=bool(
                    action_data.get("add_deprecation_header", False)
                ),
                reasoning=str(action_data.get("reasoning", ""))[:200],
            )
        except Exception as exc:
            print(f"  [WARN] Action validation failed: {exc}", flush=True)
            action = ContractAction(action_type="submit", target_field="_")

        result = env.step(action)
        obs = result.observation
        final_reward = float(result.reward or 0.0)

        log_step(
            step=step,
            action_type=action.action_type,
            target_field=action.target_field,
            reward=final_reward,
            done=result.done,
            backward=obs.backward_compat_score,
            forward=obs.forward_compat_score,
        )

        # Brief pause — stay within Gemini free-tier rate limits
        time.sleep(0.8)

    log_end(task_id, final_reward, step)
    return final_reward


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    scores: Dict[str, float] = {}

    env_client = ApiContractEnv(base_url=ENV_BASE_URL).sync()

    with env_client:
        for task_id in TASKS_TO_RUN:
            scores[task_id] = run_episode(task_id, env_client)

    elapsed = time.time() - t_start

    print("\n" + "=" * 50, flush=True)
    print("FINAL SCORES", flush=True)
    print("=" * 50, flush=True)
    for tid, score in scores.items():
        print(f"  {tid}: {score:.4f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Average : {avg:.4f}", flush=True)
    print(f"  Elapsed : {elapsed:.1f}s", flush=True)

    if elapsed >= 1200:
        raise RuntimeError(f"Exceeded 20-min limit: {elapsed:.1f}s")


if __name__ == "__main__":
    main()