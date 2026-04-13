---
title: api_contract_env
colorFrom: blue
colorTo: indigo
sdk: docker
---

#  API Contract Negotiator  OpenEnv Environment

**Meta  PyTorch  Hugging Face OpenEnv Hackathon  Round 1 Submission**

An intelligent agent learns to **evolve REST API schemas safely**satisfying new product requirements **without breaking existing consumers**. Real-world problem, real-world difficulty, real-world payoff.

## The Problem

API backward-compatibility is one of the hardest problems in production systems:

- Breaking a consumer = lost customers, emergency patches, angry engineers
- Yet APIs must evolve: new features, new data, performance improvements
- This environment teaches agents the discipline of **safe evolution**a skill no existing OpenEnv covers

## What the Agent Does

Each episode, the agent:

1.  **Observes** current schema, consumer test status, and partial scores
2.  **Proposes** one mutation per step (add field, rename, alias, change type, mark deprecated, submit)
3.  **Validates** the mutation against all consumer test suites
4.  **Receives** scored feedback with backward-compat, forward-compat, and redundancy signals
5.  **Submits** when satisfied, triggering final grading

---

##  Quick Start (30 seconds)

```bash
# Install dependencies
pip install openenv-core fastapi uvicorn pydantic openai

# Run locally (auto-starts server on :8000)
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash
export HF_TOKEN=<your-gemini-api-key>
python inference.py
```

**Expected output:**

```
[START] task_id=TASK-01 difficulty=easy max_steps=6
[STEP] step=1 action=add_field target=currency reward=0.3000 done=false ...
[STEP] step=2 action=submit target=_ reward=0.9999 done=true ...
[END] task_id=TASK-01 final_reward=0.9999 steps=2
... (TASK-02 and TASK-03 follow)
```

---

##  The Three Tasks: Easy  Medium  Hard

| Task    | Difficulty | Scenario                                                                                                                                | Max steps |
| ------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| TASK-01 | Easy       | **Add a missing field**  Add `currency` to a payments API without breaking 2 consumers                                                 | 6         |
| TASK-02 | Medium     | **Rename with backward-compat**  Rename `user_name`  `username`; keep alias for 3 legacy consumers; earn bonus for deprecation header | 10        |
| TASK-03 | Hard       | **Merge conflicting versions**  Reconcile Team A v1 + Team B v2 into one schema satisfying all 4 consumers                             | 15        |

---

##  Action Space (What the Agent Can Do)

The agent proposes one mutation per step:

```python
ContractAction(
    action_type: Literal[
        "add_field",       #  Add a new optional field
        "rename_field",    #   Rename + auto-add backward-compat alias
        "remove_field",    #  Remove (risky  causes hard breaks; use rarely)
        "add_alias",       #  Add alias pointing to existing field
        "change_type",     #  Change a field's type (e.g., int  string)
        "mark_deprecated", #   Soft-flag a field (soft removal, not hard)
        "submit",          #  Finalize and trigger full grading
    ],
    target_field: str,                  # field to act on
    new_name:     Optional[str],        # for rename/alias actions
    new_type:     Optional[str],        # for change_type action
    add_deprecation_header: bool,       #  bonus reward on Task-02 only
    reasoning:    str,                  # logged, not scored
)
```

---

##  What the Agent Observes

After each action:

```python
ContractObservation(
    done:                    bool,
    reward:                  float,          # 0.0  1.0 (step reward)
    current_schema:          List[Dict],     # live schema state
    backward_compat_score:   float,          # Can old consumers still work?
    forward_compat_score:    float,          # Can new consumers work?
    no_redundancy_score:     float,          # Clean schema (no dead fields)?
    consumer_results:        List[Dict],     # / status per consumer
    deprecation_header_present: bool,
    task_id:                 str,
    task_difficulty:         Literal["easy","medium","hard"],
    steps_remaining:         int,
    hint:                    str,            # guidance (optional)
)
```

---

##  Unique Design Features

 **Structured Logging**  `[START]`/`[STEP]`/`[END]` format is auto-parsed by the hackathon evaluator  
 **Seed-Based Reproducibility**  `reset(seed=42)` produces identical episodes every run  
 **Per-Consumer Causality**  Observation shows _exactly which consumer_ lost access to _which field_  
 **Step-Level Reward Shaping**  Intermediate scores enable hierarchical learning  
 **Parallel Sessions**  4 concurrent WebSocket connections supported (`SUPPORTS_CONCURRENT_SESSIONS = True`)  
 **Industry Best-Practices**  Deprecation header bonus teaches soft removal over hard deletion  
 **Fail-Safe Termination**  Catastrophic failures (5 consumer breakages) trigger early stop

---

##  Reward Design

**At every intermediate step:**

```
reward = backward_compat  0.30
```

**On submit (or step-limit):**

```
reward = backward_compat   0.50
       + forward_compat    0.30
       + no_redundancy     0.20
       - 0.10  hard_breaks_total       (floor 0.0)
       + 0.05 bonus if deprecation_header (TASK-02 only)
```

This design incentivizes **incremental progress** while maximizing **final quality**.

---

##  Baseline Performance

Tested with `gemini-flash-lite-latest` (Gemini API):

| Task        | Difficulty | Score      | Steps        | Result                                       |
| ----------- | ---------- | ---------- | ------------ | -------------------------------------------- |
| TASK-01     | Easy       | 0.9999     | 2            |  All consumers pass                        |
| TASK-02     | Medium     | 0.9999     | 2            |  Bonus achieved; deprecation header active |
| TASK-03     | Hard       | 0.9999     | 6            |  All 4 consumers merged successfully       |
| **Average** |           | **0.9999** | **10 total** |  Robust across all difficulties            |

All runs use `seed=42` for reproducibilityidentical scores on every submission.

---

##  Deployment Readiness

**Robustness:**

- 30-second timeouts + automatic retry logic for LLM calls
- JSON parsing with markdown fence recovery (regex fallback if model output wraps response)
- Semantic validation: invalid types, missing fields, empty strings  auto-normalized to safe `submit` action
- Task scores clamped to open interval (0, 1)  no boundary violations

**Performance:**

- Server boots in <2 seconds; typical full run = 4050 seconds
- Supports 4 concurrent WebSocket sessions
- Max 45 steps total (15 per task  3 tasks)

**Observable Correctness:**

- Structured `[START]`/`[STEP]`/`[END]` logs auto-parsed by hackathon evaluator
- No edge-case failures at score boundaries
- Early termination on catastrophic failure (5 consumer breakages)

---

##  Research Potential

This environment opens new RL research directions:

1. **Multi-Stakeholder Decision Making**  Per-consumer tracing (which stakeholder loses access to which field?) is foundational for auditable, multi-party RL.

2. **Real-World Strategy Transfer**  Agents learn aliasing, deprecation headers, and version mergingstrategies that directly apply to production API evolution.

3. **Cooperative Multi-Agent**  TASK-03 frames schema merging as negotiation between competing team interests, enabling research into distributed/cooperative RL.

4. **Advanced Reward Shaping**  Step-level partial scores (backward/forward/redundancy) enable agents to reason about trade-offs in real time.

---

##  Setup & Deployment

### Local Development (Quick Test)

```bash
# 1. Install dependencies
pip install openenv-core fastapi uvicorn pydantic openai

# 2. Run inference (server auto-starts on :8000)
export API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
export MODEL_NAME=gemini-2.0-flash
export HF_TOKEN=<your-gemini-api-key>
cd api_contract_env
python inference.py
```

Expected runtime: **4050 seconds** 

### Docker (Recommended for Production / HF Spaces)

```bash
# Build image
docker build -f server/Dockerfile -t api-contract-env:latest .

# Run container
docker run -p 8000:8000 \
  -e API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
  -e MODEL_NAME=gemini-2.0-flash \
  -e HF_TOKEN=<your-gemini-api-key> \
  api-contract-env:latest
```

### One-Command Deployment to HF Spaces

```bash
# Validate environment
openenv validate --verbose

# Deploy (auto-builds Docker image and pushes to HF Spaces)
openenv push
```

---

##  Project Structure

```
api_contract_env/
 inference.py              #  Baseline agent using Gemini LLM
 models.py                 #  Pydantic data contracts
 client.py                 #  Typed WebSocket client
 openenv.yaml              #   Environment manifest
 pyproject.toml            #  Python dependencies
 README.md                 #  Documentation

 server/
     app.py                #  FastAPI HTTP/WebSocket server
     environment.py        #  Core environment logic + 3 tasks
     Dockerfile            #  Container image
     __init__.py
```

---

##  Configuration

| Variable       | Required    | Example                                                    | Purpose                               |
| -------------- | ----------- | ---------------------------------------------------------- | ------------------------------------- |
| `API_BASE_URL` |  Yes      | `https://generativelanguage.googleapis.com/v1beta/openai/` | LLM API endpoint                      |
| `MODEL_NAME`   |  Yes      | `gemini-2.0-flash`                                         | Model identifier                      |
| `HF_TOKEN`     |  Yes      | `sk-` (Gemini key) or `hf_` (HF token)                   | API credentials                       |
| `ENV_BASE_URL` |  Optional | `http://localhost:8000`                                    | Server URL (auto-detected if not set) |

---

##  Sample Output

```
[START] task_id=TASK-01 difficulty=easy max_steps=6
[STEP] step=1 action=add_field target=currency reward=0.3000 done=false backward_compat=1.0000 forward_compat=0.0000
[STEP] step=2 action=submit target=_ reward=0.9999 done=true backward_compat=1.0000 forward_compat=1.0000
[END] task_id=TASK-01 final_reward=0.9999 steps=2

[START] task_id=TASK-02 difficulty=medium max_steps=10
[STEP] step=1 action=rename_field target=user_name reward=0.3000 done=false backward_compat=1.0000 forward_compat=0.5000
[STEP] step=2 action=submit target=_ reward=0.9999 done=true backward_compat=1.0000 forward_compat=1.0000
[END] task_id=TASK-02 final_reward=0.9999 steps=2

... TASK-03 runs for ~6 steps and also scores 0.9999
```

---

##  Who Should Use This?

| Role                     | Interest                                            | Use Case                                                             |
| ------------------------ | --------------------------------------------------- | -------------------------------------------------------------------- |
| **ML Researchers**       | Novel RL problem with multi-stakeholder constraints | Publish papers on hierarchical RL, multi-stakeholder decision-making |
| **Production Engineers** | Real API versioning scenarios                       | Learn best practices for backward-compatible schema evolution        |
| **OpenEnv Contributors** | Benchmark design                                    | Reference implementation for complex, real-world environments        |

---

##  More Information

- **OpenEnv Documentation**: [openenv on GitHub](https://github.com/meta/openenv)
- **This Repository**: [api_contract_env on GitHub](https://github.com/vigneshsiva11/Meta-env)
- **HF Spaces**: [api_contract_env on Hugging Face Spaces](https://huggingface.co/spaces/vignesh111006/api_contract_env)

---

## License

[MIT License](LICENSE)  Open source and free to use, modify, and redistribute.

---

## Author

Vignesh  B.E. Information Science and Engineering, Bannari Amman Institute of Technology  
GitHub: [vigneshsiva11](https://github.com/vigneshsiva11)
