# BitRouter Autoresearch

Benchmark harness for evaluating whether BitRouter's intelligent routing produces better task outcomes than direct API calls, using [OpenClaw](https://github.com/openclaw/openclaw) as the agent harness.

Inspired by [Karpathy's Autoresearch](https://github.com/karpathy/autoresearch) and [τ-bench](https://github.com/sierra-research/tau2-bench).

## How It Works

```
                    ┌─────────────┐
                    │  TASK.md    │
                    │  (scenario) │
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │       UserAgent         │
              │  (LLM user simulator,   │
              │   progressive disclosure)│
              └────────────┬────────────┘
                           │ multi-turn
              ┌────────────▼────────────┐
              │        OpenClaw         │
              │    (agent harness)      │
              └────────────┬────────────┘
                           │
            ┌──────────────┼──────────────┐
            │              │              │
    ┌───────▼───────┐     OR    ┌────────▼────────┐
    │   BitRouter   │           │    BitRouter     │
    │  (auto route) │           │  (passthrough)   │
    │  Condition A  │           │  → Opus 4.6      │
    └───────────────┘           │  Condition B     │
                                └─────────────────┘
                           │
              ┌────────────▼────────────┐
              │     Hybrid Judge        │
              │  programmatic + LLM     │
              └─────────────────────────┘
```

**Condition A** (bitrouter/auto): OpenClaw → BitRouter with full routing, model selection, fallback, caching.

**Condition B** (direct opus): OpenClaw → BitRouter in passthrough mode, locked to `claude-opus-4-6`.

Both conditions go through BitRouter so cost metering is identical. The only variable is routing logic.

## Quick Start

```bash
# Install
git clone git@github.com:bitrouter/bitrouter-autoresearch.git
cd bitrouter-autoresearch
uv sync

# List available tasks
uv run bench list-tasks

# Validate task files parse correctly
uv run bench validate-tasks

# Run a single trial (requires BitRouter + OpenClaw running)
uv run bench run --task easy/create-hello-file --condition bitrouter_auto --repeats 1

# Run with live API for website observation
uv run bench run --live --port 8788

# View results
uv run bench results

# Start API server (read-only, for browsing historical results)
uv run bench serve --port 8788
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [BitRouter](https://github.com/bitrouter/bitrouter) running at `localhost:8787`
- [OpenClaw](https://github.com/openclaw/openclaw) installed (`npm install -g openclaw@latest`)
- `ANTHROPIC_API_KEY` set (for UserAgent and Judge LLM calls)

## Task Format

Tasks follow a SKILL.md-adapted format with YAML frontmatter and markdown sections:

```
tasks/
├── easy/
│   └── create-hello-file/
│       └── TASK.md
├── medium/
│   └── web-research-flights/
│       └── TASK.md
└── hard/
    └── multi-step-debug/
        └── TASK.md
```

Each `TASK.md` has two isolated sections:

- **User Scenario** (given to UserAgent): persona, known info, unknown info, instructions
- **Evaluation Criteria** (given to Judge only): programmatic assertions, LLM judge rubric

The UserAgent never sees evaluation criteria. The Judge never sees condition labels (blind evaluation).

```markdown
---
name: create-hello-file
description: Create a text file with specific content.
metadata:
  difficulty: easy
  category: file_management
  budget_usd: "0.05"
  max_turns: 10
  version: "1.0"
---

# User Scenario

## Persona
Non-technical user, concise.

## Known Info
I need a file called hello.txt with "Hello World" in my home directory.

## Unknown Info
I also want a timestamp on the second line, but only if asked.

## Instructions
Ask the assistant to create the file. Say STOP when done.

---

# Evaluation Criteria

## Programmatic Assertions
- `test -f $BENCH_WORKSPACE/hello.txt`
- `grep -q "Hello World" $BENCH_WORKSPACE/hello.txt`

## LLM Judge
Rate interaction quality: Was the assistant efficient? Did it confirm success?
```

## Scoring

Hybrid evaluation with three dimensions:

| Dimension | Weight | Method |
|---|---|---|
| Task completion | 50% | Programmatic assertions (shell commands that must exit 0) |
| Interaction quality | 30% | LLM-as-judge (Claude Sonnet, blind) |
| Resource efficiency | 20% | Formula: `max(0, 1 - actual_cost / budget)` |

**Composite = 0.5 × completion + 0.3 × interaction + 0.2 × efficiency**

## Live API

Run `bench run --live` or `bench serve` to start a read-only API:

| Endpoint | Description |
|---|---|
| `GET /api/tasks` | List all benchmark tasks |
| `GET /api/trials` | List completed trials with verdicts |
| `GET /api/trials/{id}/trajectory` | Full conversation transcript |
| `GET /api/trials/live/stream` | SSE: real-time turn-by-turn stream |
| `GET /api/live/status` | Is a trial currently running? |
| `GET /api/results` | Aggregate scored results |
| `GET /api/results/compare` | Condition A vs B comparison |

SSE events on `/api/trials/live/stream`:

```
event: turn     → {"role": "user", "content": "..."}
event: turn     → {"role": "agent", "content": "..."}
event: cost     → {"cumulative_usd": 0.003, "budget_usd": 0.05}
event: done     → {"stop_reason": "user_stop", "total_cost_usd": 0.004}
```

## Configuration

Edit `configs/bench.yaml`:

```yaml
bitrouter_url: "http://localhost:8787"
conditions: [bitrouter_auto, direct_opus]
repeats: 3
budget_easy: 0.05
budget_medium: 0.10
budget_hard: 0.20
judge_model: "claude-sonnet-4-6-20250514"
user_agent_model: "claude-sonnet-4-6-20250514"
```

## Project Structure

```
bitrouter-autoresearch/
├── src/bitrouter_bench/
│   ├── runner.py        # Trial runner + orchestrator
│   ├── user_agent.py    # LLM user simulator (τ-bench style)
│   ├── judge.py         # Hybrid eval: programmatic + LLM judge
│   ├── openclaw.py      # OpenClaw subprocess interface
│   ├── cost_meter.py    # BitRouter /v1/metrics cost tracking
│   ├── trajectory.py    # JSONL trajectory storage
│   ├── events.py        # AsyncIO event bus for live streaming
│   ├── api.py           # FastAPI read-only API + SSE
│   ├── task_loader.py   # TASK.md parser
│   ├── config.py        # Configuration
│   └── cli.py           # CLI entry point
├── tasks/               # Benchmark task definitions
├── configs/             # OpenClaw condition configs + bench.yaml
├── results/runs/        # Git-tracked trial trajectories
└── analysis/            # Statistical comparison
```

## License

Apache-2.0
