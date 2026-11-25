# Deploying and operating the stack

Two compose files describe the same services:

| file | purpose |
| --- | --- |
| `docker-compose.yml` | development: source bind-mounted, `uvicorn --reload`, ports published, plus a `frontend` service running Vite with HMR |
| `docker-compose.prod.yml` | production overlay: gunicorn, nginx TLS, Postgres/Redis internal only |

```bash
cp .env.example .env          # edit POSTGRES_PASSWORD and the model backend
mkdir -p nginx/certs          # fullchain.pem + privkey.pem (see TLS below)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
curl -k https://localhost/api/ready
```

The image runs as UID **10001** (non-root) and builds the React bundle from
`frontend/` in a separate node stage, so `/` serves the UI instead of "UI not
built". The dev stack keeps that bundle by mounting an anonymous volume over
`/app/webapp/static/ui`: the source bind mount would otherwise hide it and every
fresh clone would answer 503.

For UI work use the dev server, which is the only container in this stack that
is not part of a deployment:

```bash
docker compose up -d frontend        # http://localhost:5173, HMR, /api proxied
```

## First run

0. Set `POSTGRES_PASSWORD` **before the very first start**. The official
   postgres image only applies it when it initialises an empty data directory;
   changing it later does not change the role's password, and the app will fail
   authentication with a confusing "password authentication failed" in the logs.
   If the volume already exists with a different password, either point the app
   at the existing one or `ALTER ROLE` inside the container.
1. Open `https://<host>/` and **create the first account**: it becomes the
   administrator (`role='admin'`). Everyone after it is a normal user.
2. Set `SYSML_ALLOW_SIGNUP=0` and re-deploy once your users exist.
3. Pick the provider/model in the settings panel. That configuration is stored in
   Postgres (`app_settings`), so every worker and every restart agree on it.
4. `GET /api/ready` must answer 200 before you put it behind a load balancer.

## Environment

Everything below is read from `.env` by compose; the same names work if you run
the app by hand.

| variable | default | meaning |
| --- | --- | --- |
| `DATABASE_URL` | — (required) | Postgres DSN; the overlay derives it from `POSTGRES_*` |
| `REDIS_URL` | — (required) | Redis DSN: rate limits and the per-user model cache |
| `SYSML_WORKERS` | `2` | gunicorn processes; each loads its own RAG stores |
| `SYSML_MAX_CONCURRENCY` | `4` | turns executing at once **per worker** (also the thread-pool size) |
| `SYSML_MAX_QUEUE` | `4` | turns allowed to wait behind them; beyond that the turn answers "busy" |
| `SYSML_QUEUE_WAIT` | `10` | seconds a queued turn waits for a slot |
| `SYSML_TURN_TIMEOUT` | `300` | hard deadline per turn; the stream ends with an error event |
| `SYSML_RATE_CHAT` | `20/300` | per-account chat budget (`count/window-seconds`) |
| `SYSML_RATE_AUTH` | `10/900` | failed sign-ins per address+email before the lockout |
| `SYSML_RATE_UPLOAD` | `30/3600` | uploads per account |
| `SYSML_ALLOW_SIGNUP` | `1` | `0` closes registration once an admin exists |
| `SYSML_COOKIE_SECURE` | `0` dev, `1` prod | `Secure` on the session cookie — requires HTTPS |
| `SYSML_TRUST_PROXY` | `0` dev, `1` prod | believe `X-Forwarded-For` (set by nginx) |
| `SYSML_ALLOWED_ORIGINS` | — | extra origins allowed to POST (same-host is always allowed) |
| `SYSML_ENABLE_RAG` | `1` | `0` skips the embedding model and the RAG store |
| `SYSML_SMOKE` | `0` | `1` answers every turn with a canned stream (smoke/load tests only) |
| `SYSML_SECRET_KEY` | - | required to save and encrypt per-account API keys |
| `SYSML_ALLOW_PRIVATE_ENDPOINTS` | `0` | `1` allows a saved base url that resolves to loopback/RFC1918 |
| `SYSML_STATE_DIR` | `/workspace/state` | advisory-lock file and legacy import root |
| `SYSML_LOG_LEVEL` | `INFO` | JSON log level |

## Per-account provider configuration

Every signed-in account can point its **own** turns at its own backend — a local
Ollama, an OpenAI-compatible gateway, a paid key — from the settings dialog
(`GET`/`PUT`/`DELETE /api/provider-config`). The row lives in
`user_provider_config` and layers over the deployment default above; clearing it
is how an account goes back to that default.

- **Settings → Your API keys** stores one key per account and service for
  OpenAI, OpenRouter, Anthropic, Ollama, and Tavily. The session identifies the
  owner. The API returns only which keys are saved.
- Model requests use the selected account provider's key. Switching providers or
  returning to the deployment default keeps the keys available for later use.
  Tavily web search requires the signed-in account's own Tavily key.
- Keys are encrypted with `SYSML_SECRET_KEY`. Set a stable value before users
  save keys. Operator credentials for the managed deployment backend and GitHub
  corpus ingestion remain server configuration.
- A saved base url is validated when it is written: `http(s)`, no embedded
  credentials, and no address resolving to loopback, RFC1918, link-local or the
  metadata endpoint unless `SYSML_ALLOW_PRIVATE_ENDPOINTS=1`. The check resolves
  DNS once, so a host that re-resolves later is not pinned.
- One workflow is cached per distinct backend (up to four per worker). The RAG
  pipeline, approved-solutions store and embeddings are built once and shared,
  so a second account's endpoint costs an agent and a graph — not another ~1 GB.

## Sizing

Each worker loads the embedding model and opens two Chroma stores:

| configuration | memory / worker | notes |
| --- | --- | --- |
| `SYSML_ENABLE_RAG=1` (default) | ~1 GB | full retrieval over the corpus |
| `SYSML_ENABLE_RAG=0` | ~250 MB | chat + generation only |

nginx resolves the app service name when it starts and round-robins over the
addresses it got, so **restart `web` after changing the number of replicas**
(`docker compose ... up -d --scale sysml_agent_system=3 && docker compose ... restart web`).

`2 workers × 1 GB` plus the kernel (2 GB) is the default footprint; the prod
overlay caps the app container at 6 GB. **Do not raise `SYSML_WORKERS` past
`~4` without moving the vector store out of the container** — every worker opens
the same SQLite-backed Chroma files, serialised by an `flock` (see the known
limitations below). Scaling out horizontally (`--scale app=N`) is safe for the
Postgres-backed data; the vector store is the limit.

## Backups

Postgres is the source of truth for accounts, sessions, threads, messages,
documents and backend settings.

```bash
# dump (compose network, no published port in production)
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
    exec -T postgres pg_dump -U sysml sysml | gzip > backup-$(date +%F).sql.gz

# restore into a fresh stack
gunzip -c backup-2026-01-31.sql.gz | \
    docker compose -f docker-compose.yml -f docker-compose.prod.yml \
    exec -T postgres psql -U sysml -d sysml
```

The generated models and uploads live on the `agent_workspace` volume. The two
knowledge stores are separate volumes, because `server.py` writes them under its
own `rag/` directory and the image excludes them — without those mounts every
deploy would start with no approved solutions and rebuild the index:

| volume | contents |
| --- | --- |
| `agent_workspace` | `/workspace`: per-user models and uploads |
| `chroma_data` | `/app/rag/chroma_db`: retrieval index over `rag/documents` |
| `approved_solutions` | `/app/rag/approved_solutions`: approved models and their entries |

```bash
for v in agent_workspace chroma_data approved_solutions; do
    docker run --rm -v "sysmlv2-codegeneration_agenticworkflow_$v:/data" \
        -v "$PWD":/backup alpine tar czf "/backup/$v-$(date +%F).tar.gz" -C /data .
done
```

`chroma_data` is a derived index and can be rebuilt from `rag/documents`;
`approved_solutions` cannot and must be backed up with Postgres.

Run `python -m webapp.db --check` after a restore: it applies any pending
migration and prints what is recorded.

## TLS

nginx terminates TLS on 443 and redirects 80. Drop two files into
`nginx/certs/` (they are mounted read-only and are git-ignored):

- `fullchain.pem`
- `privkey.pem`

Self-signed, for a LAN deployment:

```bash
openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
    -keyout nginx/certs/privkey.pem -out nginx/certs/fullchain.pem \
    -subj "/CN=sysml.example.internal"
```

With certbot, keep the HTTP-01 path open (it is: `/.well-known/acme-challenge/`
is served from `/var/www/certbot`) and copy the issued files into place, then
`docker compose ... restart web`.

A `Secure` cookie is dropped by the browser over plain HTTP: **leave
`SYSML_COOKIE_SECURE=0` in development**, or sign-in will appear to succeed and
then bounce straight back to the form.

## Logs

Every line is one JSON object with a `request_id`; nginx propagates it, and a
500 is answered with that id so a user can quote it.

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f app
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs -f web
```

To follow one request: `logs app | grep <request_id>`.

## Health checks

- `GET /api/health` — liveness. Cheap: no workflow build, no database round trip.
- `GET /api/ready` — readiness. Checks Postgres (`SELECT 1`), Redis (`PING`), a
  TCP connect to the kernel and the knowledge-base directories. Any failure
  answers 503 and names the component; that is what the container healthcheck,
  nginx and any scheduler should act on.

## Importing the pre-accounts state

Threads used to be JSON files keyed by a browser cookie. They are not read any
more. Re-home them under an account (never deletes anything):

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec app \
    python scripts/import_state.py --state-dir /workspace/state \
    --user-email you@example.com --dry-run
```

Drop `--dry-run` to write. The account must exist first; the importer refuses to
create one.

## Verification you can run

```bash
pytest -q                                                    # auth, isolation, limits, concurrency
python scripts/smoke.py --base-url https://localhost -k      # two accounts, isolation, approvals
# fifty accounts from one address exceed the production sign-up budget, so the
# load run uses the widened budgets in docker-compose.loadtest.yml
sed 's#rate=10r/m#rate=600r/m#' nginx/nginx.conf > nginx/nginx.loadtest.conf
docker compose -f docker-compose.yml -f docker-compose.prod.yml -f docker-compose.loadtest.yml \
  up -d --wait --scale sysml_agent_system=2
python scripts/load_test.py --base-url https://localhost -k --users 50 --duration 120
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --wait   # production budgets
```

## Known limitations

- **Web agent filesystem.** The container user (UID 10001) needs write access to
  `/workspace/users`. Each account sees its own `/uploads`, `/current` and
  `/artifacts` through virtual paths. The web agent uses file tools and reads
  the bundled reference skills.
- **Chroma on a shared volume.** The RAG and approved-solutions stores are
  SQLite files behind a `flock`. That is correct on one host and serialises
  writers, but it is the reason not to scale workers indefinitely; a dedicated
  RAG service is the follow-up.
- **Redis is required at runtime** (it is in the compose stack), but rate
  limiting fails *open*: if Redis is down, requests are allowed and the failure
  is logged rather than the app refusing traffic.
- **The kernel is internal-only.** It executes generated model code, so the prod
  overlay publishes no port for it.
- **Cold start is not free.** A freshly recreated app container that receives a
  burst of simultaneous first chats serves them only after a shared one-time
  initialisation completes; measured at 50 concurrent first requests it was
  about two minutes, with every request then completing at once. A single warm
  request afterwards answers in ~0.1 s. Warm each replica (one chat) after a
  deploy, or accept a slow first minute, before sending real traffic.
- **`--scale` is how replicas are set.** Plain `up -d` resets the app to one
  replica; use `up -d --wait --scale sysml_agent_system=2` to run two behind
  nginx.
