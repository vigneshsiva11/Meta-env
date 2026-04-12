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
• Starts environment server internally (no external dep)    ✓

Gemini OpenAI-compatible endpoint
----------------------------------
  API_BASE_URL = https://generativelanguage.googleapis.com/v1beta/openai/
  MODEL_NAME   = gemini-2.0-flash
  HF_TOKEN     = <your Gemini API key from Google AI Studio>
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
import socket
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── required env vars ─────────────────────────────────────────────────────────
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL_NAME:   str = os.environ.get("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN:     str = os.environ["HF_TOKEN"]
ENV_PORT:     int = int(os.environ.get("ENV_PORT", "8000"))
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", f"http://localhost:{ENV_PORT}")
FALLBACK_MODELS: str = os.environ.get("FALLBACK_MODELS", "gemini-flash-lite-latest,gemini-2.0-flash")

# Keep server and client on the same port when ENV_BASE_URL is explicitly provided.
try:
    parsed = urlparse(ENV_BASE_URL)
    if parsed.port is not None:
        ENV_PORT = parsed.port
except Exception:
    pass

# ── Gemini via OpenAI-compat client ──────────────────────────────────────────
llm = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
    timeout=30.0,
    max_retries=2,
)

TASKS_TO_RUN: List[str] = ["TASK-01", "TASK-02", "TASK-03"]
REWARD_MIN: float = 0.0001
REWARD_MAX: float = 0.9999


def _model_candidates() -> List[str]:
    models: List[str] = []
    for name in [MODEL_NAME, *[m.strip() for m in FALLBACK_MODELS.split(",") if m.strip()]]:
        if name not in models:
            models.append(name)
    return models


def _clamp_open_reward(score: float) -> float:
    return min(REWARD_MAX, max(REWARD_MIN, float(score)))


def _display_open_score(score: float) -> float:
    """Clamp score for stdout fields so logs never show exact 0.0 or 1.0."""
    try:
        return _clamp_open_reward(float(score))
    except Exception:
        return REWARD_MIN


# ── Server startup ────────────────────────────────────────────────────────────

def _is_port_open(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except OSError:
        return False


def start_server_background(port: int) -> None:
    import uvicorn
    from server.app import app

    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)

    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    print(f"[INFO] Starting environment server on port {port}...", flush=True)
    for _ in range(30):
        if _is_port_open(port):
            print(f"[INFO] Environment server is ready on port {port}.", flush=True)
            return
        time.sleep(1)

    raise RuntimeError(f"Environment server did not start within 30 seconds on port {port}.")


# ── Structured log helpers ────────────────────────────────────────────────────

def log_start(task_id: str, difficulty: str, max_steps: int) -> None:
    print(f"[START] task_id={task_id} difficulty={difficulty} max_steps={max_steps}", flush=True)


def log_step(step: int, action_type: str, target_field: str,
             reward: float, done: bool, backward: float, forward: float) -> None:
    reward = _display_open_score(reward)
    backward = _display_open_score(backward)
    forward = _display_open_score(forward)
    print(
        f"[STEP] step={step} action={action_type} target={target_field} "
        f"reward={reward:.4f} done={str(done).lower()} "
        f"backward_compat={backward:.4f} forward_compat={forward:.4f}",
        flush=True,
    )


def log_end(task_id: str, final_reward: float, total_steps: int) -> None:
    final_reward = _display_open_score(final_reward)
    print(f"[END] task_id={task_id} final_reward={final_reward:.4f} steps={total_steps}", flush=True)


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


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """Best-effort recovery for model outputs that include fences or extra text."""
    candidate = raw_text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        parts = candidate.split("```")
        candidate = parts[1] if len(parts) > 1 else candidate
        candidate = candidate.removeprefix("json").strip()

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _is_quota_or_rate_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    tokens = ["429", "quota", "resource_exhausted", "rate limit", "rate_limit", "too many requests"]
    return any(token in msg for token in tokens)


def call_llm(obs_text: str, task_description: str) -> Optional[Dict[str, Any]]:
    raw = ""
    models = _model_candidates()

    for model_idx, model_name in enumerate(models):
        for attempt in range(2):
            try:
                response = llm.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": f"Task:\n{task_description}\n\nCurrent environment state:\n{obs_text}"},
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = _extract_json_object(raw)
                if parsed is not None:
                    return parsed
                if attempt < 1:
                    print(f"  [WARN] JSON parse error on {model_name}; retrying | raw={raw[:120]}", flush=True)
                    time.sleep(0.2)
                    continue
                print(f"  [WARN] JSON parse error on {model_name}; switching model | raw={raw[:120]}", flush=True)
                break
            except Exception as exc:
                is_quota = _is_quota_or_rate_error(exc)
                has_next_model = model_idx < (len(models) - 1)

                if is_quota and has_next_model:
                    print(f"  [WARN] Quota/rate limit on {model_name}; switching model ({model_idx + 1}/{len(models)}).", flush=True)
                    break

                if attempt < 1 and not is_quota:
                    print(f"  [WARN] LLM call failed on {model_name} (retry): {exc}", flush=True)
                    time.sleep(0.2)
                    continue

                print(f"  [WARN] LLM call failed on {model_name}: {exc}", flush=True)
                break

    print("  [WARN] All configured models failed; falling back to safe submit action.", flush=True)
    return None


def obs_to_text(obs: Any, step: int) -> str:
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
        lines.append(f"  - {field['name']} : {field.get('type', 'str')}{alias_note}{dep_note}")
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

def run_episode(task_id: str, env: Any) -> float:
    try:
        result = env.reset(task_id=task_id, seed=42)
    except Exception as exc:
        print(f"  [WARN] reset() failed for {task_id}: {exc}", flush=True)
        log_start(task_id, "unknown", 0)
        fallback = _clamp_open_reward(0.0)
        log_end(task_id, fallback, 0)
        return fallback

    obs = result.observation
    task_description = obs.hint
    difficulty = obs.task_difficulty
    max_steps  = obs.steps_remaining

    log_start(task_id, difficulty, max_steps)

    step = 0
    final_reward: float = float(result.reward or 0.0)

    while not result.done:
        step += 1
        try:
            obs_text = obs_to_text(obs, step)
            action_data = call_llm(obs_text, task_description)
        except Exception as exc:
            print(f"  [WARN] obs/llm error at step {step}: {exc}", flush=True)
            action_data = None

        if action_data is None:
            action_data = {"action_type": "submit", "target_field": "_",
                           "reasoning": "LLM unavailable — submitting current state"}

        try:
            from models import ContractAction

            valid_action_types = {
                "add_field",
                "rename_field",
                "remove_field",
                "add_alias",
                "change_type",
                "mark_deprecated",
                "submit",
            }
            action_type = str(action_data.get("action_type") or "submit")
            if action_type not in valid_action_types:
                action_type = "submit"

            target_field = str(action_data.get("target_field") or "_")
            new_name = action_data.get("new_name") or None
            new_type = action_data.get("new_type") or None

            if action_type in {"rename_field", "add_alias"} and not new_name:
                action_type = "submit"
            if action_type == "change_type" and not new_type:
                action_type = "submit"

            action = ContractAction(
                action_type=action_type,
                target_field=target_field,
                new_name=new_name,
                new_type=new_type,
                add_deprecation_header=bool(action_data.get("add_deprecation_header", False)),
                reasoning=str(action_data.get("reasoning", ""))[:200],
            )
        except Exception as exc:
            print(f"  [WARN] Action build failed: {exc}", flush=True)
            from models import ContractAction
            action = ContractAction(action_type="submit", target_field="_")

        try:
            result = env.step(action)
        except Exception as exc:
            print(f"  [WARN] step() failed: {exc}", flush=True)
            final_reward = _clamp_open_reward(final_reward)
            log_step(step, action.action_type, action.target_field, final_reward, True, REWARD_MIN, REWARD_MIN)
            log_end(task_id, final_reward, step)
            return final_reward

        obs = result.observation
        final_reward = _clamp_open_reward(float(result.reward or 0.0))

        log_step(
            step=step,
            action_type=action.action_type,
            target_field=action.target_field,
            reward=final_reward,
            done=result.done,
            backward=obs.backward_compat_score,
            forward=obs.forward_compat_score,
        )

        time.sleep(0.5)

    log_end(task_id, final_reward, step)
    return _clamp_open_reward(final_reward)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    scores: Dict[str, float] = {}

    # Start the environment server in background if not already running
    if not _is_port_open(ENV_PORT):
        start_server_background(ENV_PORT)
    else:
        print(f"[INFO] Environment server already running on port {ENV_PORT}.", flush=True)

    time.sleep(2)

    from client import ApiContractEnv

    try:
        env_client = ApiContractEnv(base_url=ENV_BASE_URL).sync()
        with env_client:
            for task_id in TASKS_TO_RUN:
                try:
                    scores[task_id] = run_episode(task_id, env_client)
                except Exception as exc:
                    print(f"  [WARN] Episode {task_id} failed: {exc}", flush=True)
                    fallback = _clamp_open_reward(0.0)
                    log_end(task_id, fallback, 0)
                    scores[task_id] = fallback
    except Exception as exc:
        print(f"  [ERROR] Could not connect to environment: {exc}", flush=True)
        for task_id in TASKS_TO_RUN:
            scores[task_id] = _clamp_open_reward(0.0)

    elapsed = time.time() - t_start

    print("\n" + "=" * 50, flush=True)
    print("FINAL SCORES", flush=True)
    print("=" * 50, flush=True)
    for tid, score in scores.items():
        print(f"  {tid}: {_display_open_score(score):.4f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else REWARD_MIN
    print(f"  Average : {_display_open_score(avg):.4f}", flush=True)
    print(f"  Elapsed : {elapsed:.1f}s", flush=True)

    if elapsed >= 1200:
        print(f"  [WARN] Exceeded 20-min limit: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()