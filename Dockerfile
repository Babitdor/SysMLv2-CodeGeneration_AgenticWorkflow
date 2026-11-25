# Two stages: the React bundle is built with node, then the Python image is
# assembled without node, npm caches or node_modules in it.

# --- UI ---------------------------------------------------------------------
FROM node:22-alpine AS ui
WORKDIR /src/frontend
# Dependency layers first: they stay cached until package-lock.json changes.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend ./
# vite.config.ts writes to ../static/ui, i.e. /src/webapp/static/ui.
RUN npm run build

# --- Runtime ----------------------------------------------------------------
FROM python:3.11-slim AS runtime

# pip retries a dropped wheel stream and gives up on a dead one, so a bad link
# fails the build in minutes instead of hanging it forever.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_RETRIES=10 \
    PIP_DEFAULT_TIMEOUT=60 \
    SYSML_STATE_DIR=/workspace/state \
    SYSML_WORKSPACE_ROOT=/workspace

# onnxruntime (sentence-transformers) needs libgomp; everything else is a wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
# The bundle the server serves at / and /static/ui: a fresh clone returns 503
# "UI not built" without this line, which is why it is part of the image.
COPY --from=ui /src/webapp/static/ui /app/webapp/static/ui

# Non-root, and the agent's sandbox volume belongs to that user.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app \
    && chmod +x /app/scripts/docker-entrypoint.sh \
    && mkdir -p /workspace/state \
    && chown -R app:app /app /workspace

USER app

EXPOSE 8000

# Readiness, not liveness: it fails while Postgres/Redis/kernel are unusable,
# which is what a scheduler and nginx should act on.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/ready',timeout=3)"]

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
