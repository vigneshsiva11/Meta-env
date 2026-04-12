---
title: api_contract_env
colorFrom: blue
colorTo: indigo
sdk: docker
---

# API Contract Negotiator — OpenEnv Environment

> **Meta × PyTorch × Hugging Face OpenEnv Hackathon — Round 1 Submission**

An AI agent learns to evolve REST API response schemas to satisfy new product
requirements **without breaking existing consumer test suites**. Each episode
presents a real-world API versioning scenario; the agent applies typed schema
mutations and receives partial reward signals at every step.

---

## Environment description

### What the agent does

The agent receives the current API schema, consumer test results, and partial
scores. It then proposes one schema mutation per step (add a field, rename with
alias, add backward-compat alias, change type, mark deprecated, or submit).
The environment validates the mutation, runs simulated consumer test suites,
and returns a scored observation.

### Why this is a genuine real-world problem

API backward-compatibility is one of the most common and costly engineering
challenges in production systems. Breaking a consumer costs real money and
engineering time. This environment teaches agents the discipline of evolving
APIs safely — a skill no existing OpenEnv environment covers.

---

## Three tasks (easy → medium → hard)

| Task    | Difficulty | Scenario                                                                                              | Max steps |
| ------- | ---------- | ----------------------------------------------------------------------------------------------------- | --------- |
| TASK-01 | Easy       | Add a missing `currency` field to a payments API without breaking 2 consumers                         | 6         |
| TASK-02 | Medium     | Rename `user_name` → `username`; keep alias for 3 legacy consumers; earn bonus for deprecation header | 10        |
| TASK-03 | Hard       | Merge two conflicting API versions (Team A v1 + Team B v2) into one schema satisfying all 4 consumers | 15        |

---

## Action space

```python
ContractAction(
    action_type: Literal[
        "add_field",       # add a new optional field
        "rename_field",    # rename + auto-add alias
        "remove_field",    # remove (risky — causes hard breaks)
        "add_alias",       # add alias pointing to existing field
        "change_type",     # change a field's type
        "mark_deprecated", # soft-flag a field
        "submit",          # finalise and trigger full grading
    ],
    target_field: str,                  # field to act on
    new_name:     Optional[str],        # for rename/alias
    new_type:     Optional[str],        # for change_type
    add_deprecation_header: bool,       # bonus reward on Task-02
    reasoning:    str,                  # logged, not scored
)
```

---

## Observation space

```python
ContractObservation(
    done:                    bool,
    reward:                  float,          # 0.0 – 1.0
    current_schema:          List[Dict],     # live schema
    backward_compat_score:   float,          # 0.0 – 1.0
    forward_compat_score:    float,          # 0.0 – 1.0
    no_redundancy_score:     float,          # 0.0 – 1.0
    consumer_results:        List[Dict],     # per-consumer breakdown
    deprecation_header_present: bool,
    task_id:                 str,
    task_difficulty:         Literal["easy","medium","hard"],
    steps_remaining:         int,
    hint:                    str,            # guidance (empty when off)
)
```

---

## Reward function

```
Intermediate (every non-final step):
  reward = backward_compat * 0.30

Final (on "submit" or step-limit):
  reward = backward_compat  * 0.50
         + forward_compat   * 0.30
         + no_redundancy    * 0.20
         - 0.10 × hard_breaks_total    (floor 0.0)
         + 0.05 bonus if deprecation header present (Task-02 only)
```

---

## Unique features

1. **Structured stdout logs** — `[START]` / `[STEP]` / `[END]` format for auto-evaluation.
2. **Seed-based reproducibility** — `reset(seed=42)` produces identical episodes every run.
3. **Per-consumer, per-field hard-break detection** — the observation tells the agent exactly which consumer lost access to which field.
4. **Step-level hint system** — hints are present at reset; intermediate hints guide learning.
5. **`SUPPORTS_CONCURRENT_SESSIONS = True`** — 4 parallel WebSocket sessions supported.
6. **Deprecation-header bonus reward** — teaches agents the industry best practice of soft removal.
7. **Force-termination on catastrophic failure** — ≥5 hard breaks ends the episode early with a low score.

---

## Setup & installation

```bash
# 1. Install OpenEnv and dependencies
pip install openenv-core fastapi uvicorn pydantic openai

# 2. Start the server locally
cd api_contract_env
uvicorn server.app:app --host 0.0.0.0 --port 8000

# 3. Run inference (in a separate terminal)
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash
export HF_TOKEN=<your-gemini-api-key>
export ENV_BASE_URL=http://localhost:8000
python inference.py
```

### Using the OpenEnv CLI

```bash
openenv init api_contract_env   # scaffold (then copy files)
openenv serve                   # start local dev server
openenv validate --verbose      # pre-submission check
openenv push                    # deploy to Hugging Face Spaces
```

---

## Docker

```bash
# Build
openenv build
# or
docker build -f server/Dockerfile -t api-contract-env:latest .

# Run
docker run -p 8000:8000 api-contract-env:latest
```

---

## Environment variables

| Variable       | Description                                               |
| -------------- | --------------------------------------------------------- |
| `API_BASE_URL` | LLM API endpoint (e.g. Gemini OpenAI-compat URL)          |
| `MODEL_NAME`   | Model identifier (e.g. `gemini-2.0-flash`)                |
| `HF_TOKEN`     | Your API key (Gemini key, Hugging Face token, etc.)       |
| `ENV_BASE_URL` | Environment server URL (default: `http://localhost:8000`) |

---

## Project structure

```
api_contract_env/
├── __init__.py
├── models.py              # ContractAction, ContractObservation, ContractState
├── client.py              # Typed EnvClient
├── inference.py           # Baseline agent (Gemini via OpenAI client)
├── openenv.yaml           # Environment manifest
├── pyproject.toml
├── README.md
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI wiring
    ├── environment.py     # Full environment logic + 3 tasks + graders
    └── Dockerfile
```

---

## Sample inference output

```
[START] task_id=TASK-01 difficulty=easy max_steps=6
[STEP] step=1 action=add_field target=currency reward=0.3000 done=false backward_compat=1.0000 forward_compat=0.0000
[STEP] step=2 action=submit target=_ reward=0.8500 done=true backward_compat=1.0000 forward_compat=1.0000
[END] task_id=TASK-01 final_reward=0.8500 steps=2

[START] task_id=TASK-02 difficulty=medium max_steps=10
[STEP] step=1 action=rename_field target=user_name reward=0.3000 done=false backward_compat=1.0000 forward_compat=0.5000
[STEP] step=2 action=submit target=_ reward=0.9500 done=true backward_compat=1.0000 forward_compat=1.0000
[END] task_id=TASK-02 final_reward=0.9500 steps=2
```

---

## Author

Vignesh — B.E. Information Science and Engineering, Bannari Amman Institute of Technology  
GitHub: [vigneshsiva11](https://github.com/vigneshsiva11)
