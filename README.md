# SysML v2 Code Generation — Agentic Workflow

Natural-language modelling requests in, validated SysML v2 out. A FastAPI service runs a
deep agent that plans a model, generates SysML v2 text, validates it against a real SysML
kernel, repairs what fails, and streams every stage back to a React chat UI. Approved models
are stored and retrieved as few-shot context for later turns.

## What it does

- **Chat with streaming stages** — one `POST /api/threads/{id}/chat` returns SSE with
  tool rows (what ran, how long, what came back), the model text, validation results and
  versioned artifacts.
- **Deterministic pipeline** — `webapp/pipeline.py` owns stage ordering, attempt numbers and
  the retry bound (1 generation + 3 repairs by default, `SYSML_MAX_ATTEMPTS`), so event order
  never depends on the model.
- **Real validation** — code is parsed and rendered by the SysML v2 kernel sidecar
  (`microservices/sysml-kernel`), not by the model grading itself.
- **Deep agent with tools and playbooks** — `agents/SysMLAgent.py` wraps
  `deepagents.create_deep_agent`; diagram-specific playbooks live in `agents/skills/`
  (block definition, internal block, activity, package, allocation, analysis case, …) and are
  mounted read-only at runtime.
- **Solution memory** — approved models go into Chroma; later turns retrieve similar
  solutions. Uploaded documents are indexed per account and injected as context.
- **Per-account credentials** — in **Settings → Your API keys**, save separate keys for
  Ollama, OpenAI, OpenRouter, Anthropic, and Tavily. Save your preferred model provider
  in Settings to use its key. Web search uses your Tavily key.
- **Multi-tenant by default** — cookies, hashed session tokens, Postgres-keyed rows,
  per-account rate limits, queue bounds and turn deadlines.

## Architecture

```
browser (React 19 / Vite, or the bundle served by FastAPI)
        │  SSE + JSON
        ▼
FastAPI  webapp/server.py ── auth, threads, documents, artifacts, limits
        │
        ▼
webapp/pipeline.py    generate ──▶ validate ──▶ repair (bounded)
        │                 │            │
        │                 │            └──▶ microservices/sysml-kernel :8888  (SysML v2 parse/render)
        │                 └──▶ agents/ deep agent + tools (webapp/agent_tools.py)
        │                          └──▶ LLM provider (ollama / openai / openrouter / anthropic)
        ├──▶ Postgres  accounts, threads, messages, documents, artifacts, provider config
        ├──▶ Redis     rate limits
        └──▶ Chroma    approved solutions + per-account document index
```

The image builds the UI bundle from `frontend/` in a node stage and bakes it into the Python
image, so one container serves both `/` and `/api`.

## Quick start

```bash
cp .env.example .env          # optional: compose runs on its own defaults without it
docker compose up -d --build
```

Open http://localhost:8000, register — **the first account becomes the administrator** — then
pick provider and model in the settings panel and send a modelling request.

| Service | URL / port | Notes |
| --- | --- | --- |
| `sysml_agent_system` | http://localhost:8000 | FastAPI + built UI bundle |
| `sysml-kernel` | http://localhost:8888 | SysML v2 validation/rendering sidecar (POST only) |
| `frontend` | http://localhost:5173 | Vite dev server with HMR; development only |
| `postgres` | 5432 | published so host `pytest` reaches the same database |
| `redis` | 6379 | rate limits |

### Frontend editing loop

`frontend/` is a React 19 + Vite + Tailwind app with its own container:

```bash
docker compose up -d frontend     # http://localhost:5173, /api proxied to the app
```

Production never runs that container: the `ui` stage of the root `Dockerfile` builds the
bundle into the image. Outside Docker: `cd frontend && npm ci && npm run dev` (proxies to
127.0.0.1:8000), plus `npm run typecheck` and `npm run lint`.

Backend without Docker:

```bash
pip install -r requirements.txt
uvicorn webapp.server:app --host 127.0.0.1 --port 8000
```

It still needs `DATABASE_URL` and `REDIS_URL` plus a reachable kernel; the compose stack is
the supported path.

## API

`GET /api/health` (liveness), `GET /api/ready` (dependencies),
`POST /api/auth/{register,login,logout}`, `GET /api/auth/me`,
`POST /api/threads/{id}/chat` (SSE), `POST /api/validate`, `POST /api/approve`,
`POST /api/config`, `GET|PUT|DELETE /api/provider-config`, `GET /openapi.json`.

Every `/api` route except the two probes needs a session cookie, and every thread, message,
uploaded document and "last model" is stored in Postgres keyed by that account — a foreign id
answers 404 instead of leaking the row. Session tokens are stored hashed, so a database dump
cannot be replayed as a login.

## Configuration

`.env.example` documents every setting; [docs/DEPLOY.md](docs/DEPLOY.md) explains the
production values, backups, worker sizing and certificates. The ones that bite:

| Variable | Why it matters |
| --- | --- |
| `SYSML_PROVIDER`, `SYSML_MODEL`, `SYSML_BASE_URL`, `SYSML_API_KEY`, `SYSML_TEMPERATURE` | deployment default backend; the UI can override per account |
| `SYSML_SECRET_KEY` | encrypts per-account keys at rest. Without it the key field is refused and only keyless endpoints save; **rotating it makes stored keys undecryptable** |
| `SYSML_ALLOW_PRIVATE_ENDPOINTS` | SSRF guard: `0` refuses loopback/RFC1918/metadata endpoints; set `1` only when a private gateway is the point |
| `SYSML_ENABLE_RAG` | `0` skips the embedding model and vector stores (~250 MB per worker instead of ~1 GB) |
| `SYSML_MAX_CONCURRENCY`, `SYSML_MAX_QUEUE`, `SYSML_QUEUE_WAIT`, `SYSML_TURN_TIMEOUT` | turns running at once per worker, how many wait, and the deadlines |
| `SYSML_RATE_CHAT`, `SYSML_RATE_AUTH`, `SYSML_RATE_UPLOAD` | `<count>/<window seconds>` budgets per account or address |
| `SYSML_COOKIE_SECURE`, `SYSML_TRUST_PROXY` | `1` needs HTTPS and a real reverse proxy; the dev stack pins both off |
| `OLLAMA_HOST` | where a local Ollama lives (`host.docker.internal` from inside the container) |

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check .
```

Tests need Postgres and Redis (the compose stack publishes both) and a throwaway database:

```bash
TEST_DATABASE_URL=postgresql://sysml:sysml@127.0.0.1:5432/sysml \
TEST_REDIS_URL=redis://127.0.0.1:6379/15 python -m pytest -q
python scripts/smoke.py --base-url http://127.0.0.1:8000   # canned-stream smoke test
python scripts/load_test.py                                # pair with docker-compose.loadtest.yml
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest, the UI typecheck/lint/build, and a compose
build with a smoke turn, plus a nightly load run.

## Project layout

```
.
├── webapp/                  FastAPI service: server, auth, threads, documents, artifacts,
│                            pipeline, validation, providers, limits, model index
├── agents/                  deep agent + query/approval agents, prompt templates,
│   └── skills/              diagram playbooks, mounted read-only at runtime
├── tools/                   SysML validator/syntax tools and the parser package
├── rag/                     ingestion into the vector stores
├── states/                  workflow state and response models
├── frontend/                React 19 + Vite + Tailwind chat UI
├── microservices/sysml-kernel/   SysML v2 kernel sidecar (own conda environment)
├── migrations/              SQL applied by `python -m webapp.db`
├── nginx/                   TLS terminator for the production overlay
├── scripts/                 entrypoint, smoke test, load test, state import
├── tests/                   pytest suite (needs Postgres + Redis)
├── docker-compose.yml       development stack
├── docker-compose.prod.yml  TLS, gunicorn workers, unpublished database
├── docker-compose.loadtest.yml   load-test overlay
├── requirements.txt / requirements-dev.txt
└── docs/DEPLOY.md           production runbook
```

## Troubleshooting

- **503 "UI not built", or the deployed UI keeps showing strings you deleted.** The source
  mount hides the bundle baked into the image, and an anonymous volume at
  `webapp/static/ui` is seeded from the image, so a plain `npm run build` changes nothing the
  browser sees. Rebuild the image and recreate:
  `docker compose build sysml_agent_system && docker compose up -d --force-recreate sysml_agent_system`.
- **Turns fail right after a fresh clone.** The app waits on kernel/Postgres/Redis health;
  `curl localhost:8000/api/ready` names the dependency that is not up. The kernel has its own
  conda environment (`microservices/sysml-kernel/environment.yml`) and is the slowest service
  to warm up.
- **"the operator must set SYSML_SECRET_KEY".** Set it in `.env` (32+ chars, e.g.
  `openssl rand -hex 32`) and recreate the app container. Everything except storing a key works
  without it.
- **A saved private endpoint is refused.** That is `SYSML_ALLOW_PRIVATE_ENDPOINTS=0`; keep it
  and use `OLLAMA_HOST` for the local model instead.
- **Memory climbs with workers.** Each gunicorn worker loads its own copy of the vector stores:
  ~1 GB with `SYSML_ENABLE_RAG=1`, ~250 MB with `0`. Lower `SYSML_WORKERS` or disable RAG for
  chat-only deployments.
- **`pytest` cannot reach the database.** Start the compose stack; it publishes 5432 and 6379
  for exactly this reason.

## References

- [SysML v2 specification](https://www.omg.org/sysml/)
- [LangGraph](https://langchain-ai.github.io/langgraph/) · [LangChain](https://python.langchain.com)
- [ChromaDB](https://docs.trychroma.com)
- [DEPLOY.md](docs/DEPLOY.md) — production overlay and runbook
