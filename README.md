# OmniAgent

**Production-ready, self-improving Python agent framework** with a polling-based control plane, closed learning loop, and ubiquitous messaging gateway.

---

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the control plane

```bash
omniagent server
# or
python -m omniagent server
```

The server starts at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive API.

### 4. Run a job locally

```bash
omniagent run-job --job-name "my_job" --agent-id "<uuid>"
```

---

## Architecture

```
omniagent/
├── core/        # Agent loop (Think-Act-Observe), memory, skills
├── api/         # FastAPI control plane
├── gateway/     # Telegram / Discord / Slack adapters
├── inference/   # LLM provider abstraction (Claude, GPT, Ollama, Groq, OpenRouter, NVIDIA NIM)
├── skills/      # Built-in + auto-synthesized skills
└── db/          # SQLAlchemy ORM + async engine
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Job** | A recurring (cron) or one-off task with context injections |
| **Run** | A single execution of a Job; captures full trajectory |
| **Skill** | Reusable Python function synthesised from a successful Run |
| **Agent** | A polling worker that pulls Jobs from the control plane |
| **OmniMessage** | Normalised message from any channel (Telegram, Slack, …) |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server liveness check |
| GET | `/api/v1/agents/{id}/next-job` | Long-poll for next pending job |
| POST | `/api/v1/runs` | Submit run result |
| GET | `/api/v1/runs/{id}` | Fetch run details |
| POST | `/api/v1/jobs` | Create a new job |
| GET | `/api/v1/jobs` | List jobs |
| GET | `/api/v1/agents` | List agents |
| POST | `/api/v1/agents` | Register agent |
| POST | `/gateway/telegram/webhook` | Telegram webhook receiver |

---

## Running Tests

```bash
pytest
```

---

## Environment Variables

See `.env.example` for the full list of supported variables.
