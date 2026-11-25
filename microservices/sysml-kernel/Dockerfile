# ── SysML Jupyter Kernel Sidecar ──────────────────────────────────────────
#
# Lightweight HTTP server that renders SysML v2 textual notation as Graphviz
# PNG diagrams using the official OMG jupyter-sysml-kernel.
#
# The backend calls this service via HTTP instead of requiring Java 21,
# Graphviz, and the kernel to be installed on the backend container.
#
# Usage:
#   docker compose up sysml-kernel
#   curl -X POST http://localhost:8888/render \
#     -H "Content-Type: application/json" \
#     -d '{"sysml_text": "package P { part def A; }", "element_name": "P"}'
#
# ── Optimization notes (2026-09-03 rebuild) ───────────────────────────────
# 1. The old image apt-installed temurin-21-jdk AND carried conda-forge
#    openjdk (pulled in by jupyter-sysml-kernel's `openjdk>=11` dep) — two
#    complete Java runtimes, one redundant (~346 MB). The Adoptium block is
#    gone; the kernelspec's `kernel.json` invokes plain `java` from PATH,
#    which conda-forge's openjdk provides.
# 2. The environment is pinned in environment.yml (solve-once, restore-
#    forever): Dockerfile edits no longer trigger a multi-minute solver run,
#    and this layer only rebuilds when the YAML changes.
# 3. `jupyter_console` was dropped — server.py only uses jupyter_client.
FROM continuumio/miniconda3:24.11.1-0

# Environment pinned in environment.yml — `conda env update` applies the
# spec to the base env instead of re-solving an unpinned request. Cache
# mount keeps downloaded packages across builds; the clean happens in the
# SAME RUN so it doesn't orphan a layer.
COPY environment.yml /tmp/environment.yml
RUN --mount=type=cache,target=/opt/conda/pkgs,sharing=locked \
    conda env update -n base -f /tmp/environment.yml && \
    rm /tmp/environment.yml && \
    conda clean -afy

WORKDIR /app

COPY server.py .

# ── Non-root ──────────────────────────────────────────────────────────────
# The renderer only writes scratch files to the system temp dir, but
# jupyter_client also needs a writable $HOME for its runtime connection
# files, hence --create-home. Nothing here requires root.
RUN useradd --create-home --uid 10001 appuser
ENV HOME=/home/appuser
USER appuser

EXPOSE 8888

# The server uses Python's built-in http.server — no extra dependencies.
CMD ["python", "server.py"]
