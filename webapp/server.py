"""FastAPI surface for the SysML v2 deep agent.

Multi-tenant by construction: every /api route except the two probes resolves the
signed-in account first (session cookie -> sessions table -> user) and every
store is keyed by that account, so a foreign thread or document id reads as
"not found" instead of opening someone else's work, and two browsers never share
a transcript, an upload directory or a "last model".

Turns are bounded: an admission gate caps how many are queued and running, the
work itself runs in a fixed thread pool, and each turn has a deadline. That is
what makes "gunicorn -w N" safe to run behind nginx — nothing per request
spawns an unbounded thread any more.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from psycopg.types.json import Jsonb
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from webapp import (  # noqa: E402
    artifacts,
    auth,
    db,
    documents,
    limits,
    locks,
    logs,
    providers,
    router,
    threads,
)
from webapp.pipeline import run_sysml_pipeline  # noqa: E402
from webapp.router import GENERAL_CHAT, SYSML_GENERATION  # noqa: E402
from webapp.tool_schemas import TOOL_MANIFEST  # noqa: E402

logger = logging.getLogger("webapp.server")

# Created before the routes (they need the decorator); the JSON log formatter,
# request ids and the startup/shutdown hook are attached at the end of the file.
app = FastAPI(title="SysML v2 Deep Agent")

STATIC_DIR = Path(__file__).resolve().parent / "static"
UI_DIR = STATIC_DIR / "ui"

# Backend selection: SYSML_PROVIDER picks the branch, SYSML_MODEL the model.
# Ollama takes a bare model name; the hosted providers take a config dict.
DEFAULT_MODELS = {
    "ollama": "glm-5.3-flash:cloud",
    "openai": "gpt-5-mini",
    "openrouter": "qwen/qwen3-coder",
    "anthropic": "claude-sonnet-4-5-20250929",
}
DEFAULT_TEMPERATURE = 0.15

# The agent's sandbox is the container filesystem; each account gets its own
# subtree so one user's prompt cannot read another's uploads by path.
STATE_DIR = Path(os.environ.get("SYSML_STATE_DIR", "/workspace/state"))
WORKSPACE_ROOT = Path(os.environ.get("SYSML_WORKSPACE_ROOT", "/workspace"))
MAX_UPLOAD_BYTES = 10_000_000  # the backend caps a single file at 10 MiB
MAX_TURNS = 12
MAX_TURN_CHARS = 4000

# Turn control. Concurrency also sizes the pool, so the two cannot disagree.
MAX_CONCURRENCY = max(1, int(os.environ.get("SYSML_MAX_CONCURRENCY") or 4))
MAX_QUEUE = max(0, int(os.environ.get("SYSML_MAX_QUEUE") or 4))
QUEUE_WAIT = float(os.environ.get("SYSML_QUEUE_WAIT") or 10.0)
TURN_TIMEOUT = float(os.environ.get("SYSML_TURN_TIMEOUT") or 300.0)
HEARTBEAT = 15.0  # SSE ping so a proxy does not close an idle stream

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

TURN_POOL = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY, thread_name_prefix="turn")

# Backend settings written by /api/config are deployment-wide, so they live in
# app_settings and every worker sees them; this short cache keeps the per-request
# cost to a dict read while still picking up another worker's write quickly.
CONFIG_LOCK_KEY = 8_274_119_006
SETTINGS_TTL = float(os.environ.get("SYSML_SETTINGS_TTL") or 2.0)
_EMPTY_SETTINGS: Dict[str, Any] = {
    "overrides": {},
    "profiles": {"active": None, "profiles": {}},
    "revision": 0.0,
}
_settings: Dict[str, Any] = dict(_EMPTY_SETTINGS, at=0.0)
_settings_lock = threading.Lock()


# --- Settings: environment plus whatever the UI saved ------------------------


def _parse_settings(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """app_settings rows -> {overrides, profiles, revision}."""
    state: Dict[str, Any] = {
        "overrides": {},
        "profiles": {"active": None, "profiles": {}},
        "revision": 0.0,
    }
    for row in rows:
        try:
            state["revision"] = max(state["revision"], float(row.get("at") or 0.0))
        except (TypeError, ValueError):
            pass
        value = row.get("value")
        if not isinstance(value, dict):
            continue
        if row["key"] == "overrides":
            state["overrides"] = {
                str(key): str(item)
                for key, item in value.items()
                if str(key).startswith("SYSML_")
                and isinstance(item, (str, int, float, bool))
            }
        elif row["key"] == "profiles":
            saved = value.get("profiles")
            profiles: Dict[str, Dict[str, str]] = {}
            if isinstance(saved, dict):
                for name, fields in saved.items():
                    if not isinstance(fields, dict):
                        continue
                    profiles[str(name)] = {
                        str(key): str(item)
                        for key, item in fields.items()
                        if str(key).startswith("SYSML_")
                        and isinstance(item, (str, int, float, bool))
                    }
            active = value.get("active")
            state["profiles"] = {
                "active": str(active) if isinstance(active, str) else None,
                "profiles": profiles,
            }
    return state


_SETTINGS_SQL = "select key, value, extract(epoch from updated_at) as at from app_settings"


def _read_settings() -> Dict[str, Any]:
    try:
        with db.connection() as conn:
            rows = conn.execute(_SETTINGS_SQL).fetchall()
    except Exception as exc:
        # A missing table (first boot) or a dead database must not take the whole
        # app down: fall back to the environment, and let /api/ready report it.
        logger.warning("settings unavailable (%s): using the environment", exc)
        return dict(_EMPTY_SETTINGS)
    return _parse_settings(rows)


def settings_snapshot(force: bool = False) -> Dict[str, Any]:
    """The cached settings, refreshed from Postgres at most every SETTINGS_TTL."""
    now = time.monotonic()
    with _settings_lock:
        if not force and _settings["at"] and now - _settings["at"] < SETTINGS_TTL:
            return _settings
    fresh = _read_settings()
    with _settings_lock:
        _settings.update(fresh, at=now)
        return _settings


def _write_settings(state: Dict[str, Any]) -> None:
    """Replace both settings rows atomically, then refresh the cache."""
    with db.connection() as conn:
        with conn.transaction():
            conn.execute("select pg_advisory_xact_lock(%s)", (CONFIG_LOCK_KEY,))
            for key, value in (
                ("overrides", state["overrides"]),
                ("profiles", state["profiles"]),
            ):
                conn.execute(
                    "insert into app_settings (key, value, updated_at) "
                    "values (%s, %s, now()) "
                    "on conflict (key) do update "
                    "set value = excluded.value, updated_at = now()",
                    (key, Jsonb(value)),
                )
    settings_snapshot(force=True)


def effective_env(user_id: Optional[str] = None) -> Dict[str, str]:
    """SYSML_* settings: environment, then the deployment, then this account.

    The deployment overrides are the administrator's `/api/config`; a user_id
    layers that account's own provider row on top, so one account can bring an
    endpoint or a key without changing what everyone else gets. Callers that
    pass no user_id (health, /api/config, reset) see the deployment view.
    """
    env = {key: value for key, value in os.environ.items() if key.startswith("SYSML_")}
    env.update(settings_snapshot()["overrides"])
    if user_id:
        env.update(providers.overrides(str(user_id)))
    return env


def active_backend(env: Optional[Dict[str, str]] = None):
    """(provider, model) after env + UI overrides."""
    env = effective_env() if env is None else env
    provider = (env.get("SYSML_PROVIDER") or "ollama").strip().lower()
    model = env.get("SYSML_MODEL") or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["ollama"])
    return provider, model


def active_temperature(env: Optional[Dict[str, str]] = None) -> float:
    env = effective_env() if env is None else env
    try:
        return float(env.get("SYSML_TEMPERATURE") or DEFAULT_TEMPERATURE)
    except ValueError:
        return DEFAULT_TEMPERATURE


def provider_config(env: Optional[Dict[str, str]] = None) -> Any:
    """LLM config for SysMLWorkflow: bare Ollama model name or provider dict.

    OpenAI/openrouter/anthropic read SYSML_BASE_URL, SYSML_API_KEY,
    SYSML_EFFORT (reasoning effort), SYSML_MAX_TOKENS and SYSML_TIMEOUT.
    Pass env to resolve a different mapping (tests do this).
    """
    env = effective_env() if env is None else env
    provider = (env.get("SYSML_PROVIDER") or "ollama").strip().lower()
    model = env.get("SYSML_MODEL") or DEFAULT_MODELS.get(provider, DEFAULT_MODELS["ollama"])
    if provider == "ollama":
        return model

    config: Dict[str, Any] = {"provider": provider, "model": model}
    for name, key in (
        ("SYSML_BASE_URL", "base_url"),
        ("SYSML_API_KEY", "api_key"),
        ("SYSML_EFFORT", "reasoning_effort"),
        ("SYSML_MAX_TOKENS", "max_completion_tokens"),
        ("SYSML_TIMEOUT", "timeout"),
    ):
        value = env.get(name)
        if not value:
            continue
        config[key] = int(value) if key in ("max_completion_tokens", "timeout") else value
    return config


def _profile_view(name: str, values: Dict[str, str]) -> Dict[str, Any]:
    """One saved configuration as the settings dialog shows it (never the key)."""
    env = dict(values)
    provider, model = active_backend(env)
    return {
        "name": name,
        "provider": provider,
        "model": model,
        "temperature": active_temperature(env),
        "base_url": env.get("SYSML_BASE_URL", ""),
        "effort": env.get("SYSML_EFFORT", ""),
        "max_tokens": env.get("SYSML_MAX_TOKENS"),
        "timeout": env.get("SYSML_TIMEOUT"),
        "api_key_set": bool(env.get("SYSML_API_KEY")),
    }


# --- Workflow ---------------------------------------------------------------
# Built per distinct backend — an account's own provider row included — because
# the agents and the graph are cheap and a rebuild is how a settings change takes
# effect. The flock around build/cleanup keeps N workers from opening the same
# SQLite-backed store at the same moment. The RAG pipeline, the approved
# solutions store and the embeddings (~1 GB) are built once by the first
# workflow and shared by the rest, so a second account's endpoint costs an agent
# and a graph rather than another copy of the embedding model.

_workflow_lock = threading.Lock()
# providers.signature(env) -> workflow; insertion order is also the LRU order.
_workflows: Dict[Any, Any] = {}
MAX_WORKFLOWS = 4
# The first workflow's stores, handed to every later one; None until it is built.
_shared: Optional[Dict[str, Any]] = None


def _shared_stores(workflow: Any = None) -> Optional[Dict[str, Any]]:
    """The heavy stack, built once by the first workflow and shared afterwards.

    Called with that first workflow it records what SysMLWorkflow built for it
    (under the flock, so the SQLite stores are opened once across workers), and
    every later workflow reuses it instead of paying for it again.
    """
    global _shared
    if workflow is not None and _shared is None:
        _shared = {
            "rag_pipeline": getattr(workflow, "rag_pipeline", None),
            "approved_solutions_db": getattr(workflow, "approved_solutions_db", None),
            "embeddings": getattr(workflow, "embeddings", None),
        }
    return _shared


def _build_workflow(env: Dict[str, str], shared: Optional[Dict[str, Any]] = None):
    from build_graph import SysMLWorkflow

    with locks.flock():
        return SysMLWorkflow(
            agents_prompt=str(ROOT / "agents" / "prompt" / "prompt.yaml"),
            common_llm=provider_config(env),
            model_name=active_backend(env)[1],
            temperature=active_temperature(env),
            # Chat-only deployments can skip the embedding model entirely
            # (~0.5-1 GB per worker, see docs/DEPLOY.md).
            enable_rag=(env.get("SYSML_ENABLE_RAG") or "1").strip().lower()
            not in ("0", "false", "no"),
            rag_persist_directory=str(ROOT / "rag" / "chroma_db"),
            approved_solutions_persist_dir=str(ROOT / "rag" / "approved_solutions"),
            **(shared or {}),
        )


def _cleanup(workflow: Any) -> None:
    if workflow is None:
        return
    try:
        with locks.flock():
            workflow.cleanup()
    except Exception:
        pass


def drop_workflow() -> None:
    """Forget the cached workflows so the next turn rebuilds with new settings."""
    with _workflow_lock:
        previous = list(_workflows.values())
        _workflows.clear()
    for workflow in previous:
        _cleanup(workflow)


def get_workflow(user_id: Optional[str] = None):
    """The workflow for this account's backend (built on first use).

    Accounts that resolve to the same backend share one workflow: the signature
    covers provider, model, endpoint, key, temperature and effort, so a
    per-account override costs a build only when it really differs. The
    deployment revision is part of the key, so an admin's change still rebuilds
    everything (max_tokens and timeout included).
    """
    env = effective_env(user_id)
    key = (providers.signature(env), settings_snapshot()["revision"])
    with _workflow_lock:
        cached = _workflows.pop(key, None)  # popped so it re-enters as newest
        if cached is not None:
            _workflows[key] = cached
            return cached
        workflow = _build_workflow(env, _shared_stores())
        _shared_stores(workflow)
        _workflows[key] = workflow
        evicted = []
        while len(_workflows) > MAX_WORKFLOWS:
            # Least recently used: the oldest entry, in insertion order.
            oldest = next(iter(_workflows))
            evicted.append(_workflows.pop(oldest))
    for stale in evicted:
        _cleanup(stale)
    return workflow


# --- Per-user workspace and last model --------------------------------------


def _owner_id(user: Dict[str, Any]) -> str:
    return str(user["id"])


def workspace_paths(user_id: str) -> Dict[str, Path]:
    """<workspace>/users/<id>/{uploads,current,artifacts}, created on demand."""
    safe = "".join(ch for ch in str(user_id) if ch.isalnum() or ch in "-_")[:64] or "unknown"
    root = WORKSPACE_ROOT / "users" / safe
    paths = {
        "root": root,
        "uploads": root / "uploads",
        "current": root / "current",
        "artifacts": root / "artifacts",
    }
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    return paths


def _model_key(user_id: str) -> str:
    return f"model:{user_id}"


def _remember_model(user_id: str, code: str) -> None:
    """Track this user's validated model: Redis for the stack, a file for restarts.

    Redis is the shared copy every worker reads, so a model generated on worker A
    draws on worker B; the file under the user's workspace is the fallback when
    Redis is unavailable and what a restart recovers from.
    """
    if not code.strip():
        return
    target = workspace_paths(user_id)["current"] / "last_model.sysml"
    tmp = target.with_suffix(".sysml.tmp")
    try:
        tmp.write_text(code, encoding="utf-8")
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except OSError:
        pass
    try:
        client = db.redis_client()
        with client.pipeline() as pipe:
            pipe.hset(_model_key(user_id), "code", code)
            pipe.hincrby(_model_key(user_id), "revision", 1)
            pipe.expire(_model_key(user_id), 30 * 24 * 3600)
            pipe.execute()
    except Exception as exc:
        logger.warning("could not cache the model in Redis (%s)", exc)


def model_state(user_id: str) -> Dict[str, Any]:
    """This user's last validated model and a revision to poll for changes."""
    code, revision = "", 0
    try:
        cached = db.redis_client().hgetall(_model_key(user_id))
        if cached:
            code = str(cached.get("code") or "")
            revision = int(cached.get("revision") or 0)
    except Exception as exc:
        logger.warning("could not read the model from Redis (%s)", exc)
    if not code:
        target = workspace_paths(user_id)["current"] / "last_model.sysml"
        try:
            code = target.read_text(encoding="utf-8")
            revision = int(target.stat().st_mtime)
        except (OSError, ValueError):
            pass
    return {"code": code, "revision": revision}


# Renders are keyed by user as well: the payload embeds a specific model, and one
# user's diagram must never be served to another. FIFO-capped so it cannot grow
# without bound in a long-lived worker.
RENDER_CACHE_MAX = 64
_render_cache: Dict[Any, Dict[str, Any]] = {}
_render_cache_lock = threading.Lock()


def _render_cached(key: Any) -> Optional[Dict[str, Any]]:
    with _render_cache_lock:
        return _render_cache.get(key)


def _render_store(key: Any, payload: Dict[str, Any]) -> None:
    with _render_cache_lock:
        if len(_render_cache) >= RENDER_CACHE_MAX:
            for stale in list(_render_cache)[: RENDER_CACHE_MAX // 2]:
                _render_cache.pop(stale, None)
        _render_cache[key] = payload


# --- SysML visualizer -------------------------------------------------------
# The kernel sidecar owns the %viz renderer. These endpoints expose it so a
# client can draw the diagram beside the model instead of in the chat stream.
_ORIENTATIONS = ("TB", "LR")
_COMPARTMENT_STYLES = ("default", "compact", "tree")


class RenderRequest(BaseModel):
    code: Optional[str] = None
    element_name: str = ""
    orientation: str = "TB"
    compartments: str = "default"


def _kernel_render(
    code: str,
    element_name: str = "",
    orientation: str = "TB",
    compartments: str = "default",
) -> Dict[str, Any]:
    """POST the model to the kernel's %viz renderer and return its JSON reply."""
    if orientation not in _ORIENTATIONS:
        raise HTTPException(status_code=400, detail="orientation must be TB or LR")
    if compartments not in _COMPARTMENT_STYLES:
        raise HTTPException(
            status_code=400, detail="compartments must be default, compact or tree"
        )
    service_url = os.getenv("SYSML_KERNEL_SERVICE_URL", "http://sysml-kernel:8888")
    payload = json.dumps(
        {
            "sysml_text": code,
            "element_name": element_name,
            "orientation": orientation,
            "compartments": compartments,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{service_url.rstrip('/')}/render",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=320) as response:
            reply = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # The kernel reports render failures as 4xx/5xx JSON; keep its message.
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except (OSError, ValueError):
            detail = {}
        message = detail.get("message") if isinstance(detail, dict) else None
        return {
            "success": False,
            "message": str(
                message or f"Visualizer service at {service_url} answered HTTP {exc.code}."
            ),
        }
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {
            "success": False,
            "message": (
                f"Visualizer service unreachable at {service_url} "
                f"({type(exc).__name__}). Start it with "
                "'docker compose up -d sysml-kernel'."
            ),
        }
    if not isinstance(reply, dict):
        return {"success": False, "message": "Malformed render response"}
    return reply


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _label() -> str:
    """provider/model tag used in user-facing errors."""
    provider, model = active_backend()
    return f"{provider}/{model}"


def _sse(event: Dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None
    doc_ids: List[str] = []
    history: List[Dict[str, str]] = []
    artifact: str = ""


class ThreadCreate(BaseModel):
    title: str = ""


class ThreadRename(BaseModel):
    title: str


class ThreadChatRequest(BaseModel):
    message: str = ""
    # Parity with ChatRequest: attached documents ground the answer and the
    # current model is fed back so "extend that" keeps working without a panel.
    doc_ids: List[str] = []
    artifact: str = ""


class ArtifactCreate(BaseModel):
    """A new canvas document; `kind` is validated against artifacts.KINDS."""

    title: str = ""
    kind: str = "code"
    content: str = ""


class ArtifactVersionIn(BaseModel):
    """One saved version of an existing artifact."""

    kind: str = "code"
    content: str = ""


class ArtifactRestore(BaseModel):
    version: int


class ApprovalRequest(BaseModel):
    code: str
    task: str = "web approval"
    feedback: str = ""


class ValidateRequest(BaseModel):
    code: str


class ModelRequest(BaseModel):
    code: str = ""


class ModelImpactRequest(ModelRequest):
    element_id: str = ""


class DocumentInfo(BaseModel):
    id: str
    name: str
    chars: int
    chunks: int


class Credentials(BaseModel):
    email: str
    password: str


_HISTORY_ROLES = {"user", "assistant"}


def _sanitize_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Last 8 turns, user/assistant only, <=4000 chars each; junk tolerated."""
    turns: List[Dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role in _HISTORY_ROLES and content:
            turns.append({"role": role, "content": content[:4000]})
    return turns[-8:]


def _history_block(turns: List[Dict[str, str]]) -> str:
    """Transcript prefix so a turn continues the same conversation.

    Derived from the stored thread rather than from process memory, which is what
    makes generation worker-independent: the checkpointer is only a cache.
    """
    if not turns:
        return ""
    lines = [
        f"{turn['role']}: {turn['text'][:MAX_TURN_CHARS]}" for turn in turns[-MAX_TURNS:]
    ]
    return "EARLIER IN THIS CONVERSATION:\n" + "\n".join(lines) + "\n\n"


MODE_INSTRUCTIONS = {
    GENERAL_CHAT: (
        "Task: answer the question or continue the conversation. Be concrete, "
        "name the document/chunk you used when documents are attached, and "
        "include SysML v2 only if it helps the explanation. "
        "Format the answer as Markdown: short paragraphs, dash bullets for lists, "
        "**bold** for key terms, and fenced code blocks with a language tag for code."
    ),
    SYSML_GENERATION: (
        "Task: produce a complete SysML v2 model that satisfies the request. Fill the "
        "'code' field of the JSON response with the full model."
    ),
}

_V1_CONVERSION_NOTE = (
    "The request references SysML v1 / UML-style input. Express the result as "
    "SysML v2 syntax (the validator only accepts v2): convert packages, part "
    "definitions and internal block diagrams to their v2 equivalents."
)


def _workspace_note(user_id: str) -> str:
    """The agent's workspace contract: its own directory, never a shared one."""
    root = workspace_paths(user_id)["root"]
    return (
        f"Workspace: the files for this conversation are in {root}/ "
        "(uploads/, current/, artifacts/). Read and write only inside that "
        "directory; a path outside it belongs to another user."
    )


NO_READABLE = "No readable content was extracted from the attached documents."


def _retrieved(
    user_id: str, doc_ids: List[str], message: str
) -> List[Dict[str, str]]:
    """Keyword-matched chunks for the attached documents, else their openings."""
    matches = documents.search(user_id, doc_ids, message)
    if not matches:  # no keyword hit: show the opening of each document
        matches = [
            {"document": d.name, "chunk": "1", "text": d.chunks[0]}
            for d in documents.list_documents(user_id)
            if d.id in doc_ids and d.chunks
        ]
    return matches


def _retrieval_event(matches: List[Dict[str, str]]) -> Dict[str, Any]:
    """The retrieval as one trace row: exactly the context the model receives."""
    # Not a cycle: agents imports tools/states, never webapp, and this module is
    # fully loaded by the time a turn runs.
    from agents.SysMLAgent import _preview

    if matches:
        text = "SOURCE DOCUMENTS:\n\n" + documents.context_block(matches)
        names = ", ".join(dict.fromkeys(m["document"] for m in matches))
        count = len(matches)
        summary = f"{count} chunk{'s' if count != 1 else ''} from {names}"
    else:
        text = NO_READABLE
        summary = "no readable content in the attached documents"
    output, truncated, chars = _preview(text)
    return {
        "type": "tool",
        "name": "retrieve_documents",
        "status": "done",
        "call_id": "retrieval",
        "summary": summary,
        "chars": chars,
        "truncated": truncated,
        "output": output,
    }


def _compose_prompt(
    user_id: str,
    message: str,
    route: str,
    matches: Optional[List[Dict[str, str]]],
    artifact: str = "",
    model_format: Optional[str] = None,
) -> str:
    """User turn request + retrieved document context + the mode contract."""
    parts = [message]
    if matches is not None:
        if matches:
            parts.append("SOURCE DOCUMENTS:\n\n" + documents.context_block(matches))
        else:
            parts.append(NO_READABLE)
    if route == SYSML_GENERATION:
        artifact = (artifact or "").strip()
        if artifact:
            parts.append(
                "CURRENT MODEL (modify or extend this, do not start from scratch):\n\n"
                + artifact
            )
        if model_format == "v1":
            parts.append(_V1_CONVERSION_NOTE)
    parts.append(MODE_INSTRUCTIONS.get(route, MODE_INSTRUCTIONS[GENERAL_CHAT]))
    parts.append(_workspace_note(user_id))
    return "\n\n".join(parts)


# --- Bounded turns ----------------------------------------------------------

SMOKE_MODEL = "package Smoke {\n    part def Ping;\n}"


def _smoke_mode() -> bool:
    """SYSML_SMOKE=1 answers every turn with a canned stream and no provider call.

    scripts/smoke.py uses it to exercise the deployment end to end — routing,
    streaming, persistence, tenant isolation — without spending model calls or
    needing a reachable provider.
    """
    return (os.environ.get("SYSML_SMOKE") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _approved_hits() -> int:
    """How many approved solutions *this worker* can retrieve (smoke mode only).

    Approving on one replica and chatting on another is the cross-worker check:
    a non-zero count on the second worker proves the store is shared and
    searchable, which is exactly what webapp/locks.py serialises.
    """
    try:
        workflow = get_workflow()
        with locks.flock():
            found = workflow.search_similar_solutions("package Smoke part def Ping")
        return len(found or [])
    except Exception as exc:
        logger.warning("smoke probe could not search the approved store (%s)", exc)
        return -1


async def _smoke_events() -> AsyncIterator[Dict[str, Any]]:
    """The canned turn: one route frame, prose, a model and a verdict."""
    yield {
        "type": "route",
        "route": SYSML_GENERATION,
        "mode": SYSML_GENERATION,
        "reason": "smoke",
        "source": "smoke",
        "format": "v2",
    }
    hits = await asyncio.to_thread(_approved_hits)
    yield {"type": "token", "text": f"approved={hits} "}
    for word in ("smoke", "test", "reply"):
        yield {"type": "token", "text": word + " "}
    yield {"type": "code", "code": SMOKE_MODEL, "attempt": 1, "final": True}
    yield {"type": "validation", "status": "valid", "ok": True, "attempt": 1}


def _pump_pipeline(agent, prompt, thread_id, loop, queue, stop) -> None:
    """Run the blocking pipeline generator off the event loop."""
    stream = None
    try:
        stream = run_sysml_pipeline(agent, prompt, thread_id)
        for event in stream:
            if stop.is_set():
                break
            loop.call_soon_threadsafe(queue.put_nowait, event)
    except Exception as exc:  # unreachable Ollama, provider error, ...
        loop.call_soon_threadsafe(
            queue.put_nowait, {"type": "error", "detail": f"{_label()}: {exc}"}
        )
    finally:
        # Closing the generator releases the agent's per-turn resources; the
        # sentinel then tells the consumer there is nothing more coming.
        closer = getattr(stream, "close", None)
        if closer is not None:
            try:
                closer()
            except Exception:
                pass
        try:
            loop.call_soon_threadsafe(queue.put_nowait, None)
        except RuntimeError:  # loop already closed (shutdown)
            pass


def _pump_chat(llm, messages, loop, queue, stop) -> None:
    """Run blocking plain-chat token streaming off the event loop."""
    try:
        for chunk in llm.stream(messages):
            if stop.is_set():
                break
            text = getattr(chunk, "text", "")
            if not isinstance(text, str):
                text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                loop.call_soon_threadsafe(
                    queue.put_nowait, {"type": "token", "text": text}
                )
    except Exception as exc:
        loop.call_soon_threadsafe(
            queue.put_nowait, {"type": "error", "detail": f"{_label()}: {exc}"}
        )
    finally:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, None)
        except RuntimeError:
            pass


_gates: Dict[str, Any] = {}


def _turn_gates():
    """Admission gates for the running loop.

    The "queue" gate bounds everything admitted (so the backlog has a ceiling)
    and "run" bounds what is actually executing; the pool has exactly its size.
    """
    loop = asyncio.get_running_loop()
    if _gates.get("loop") is not loop:
        _gates.update(
            loop=loop,
            queue=asyncio.Semaphore(MAX_CONCURRENCY + MAX_QUEUE),
            run=asyncio.Semaphore(MAX_CONCURRENCY),
        )
    return _gates["queue"], _gates["run"]


async def _admit(queue_gate, run_gate) -> bool:
    """Take both gates within QUEUE_WAIT, or give up (the caller streams 503-ish)."""

    async def take() -> None:
        await queue_gate.acquire()
        try:
            await run_gate.acquire()
        except BaseException:
            queue_gate.release()
            raise

    try:
        await asyncio.wait_for(take(), QUEUE_WAIT)
        return True
    except (asyncio.TimeoutError, TimeoutError):
        return False


async def _turn_events(
    *,
    user_id: str,
    message: str,
    doc_ids: List[str],
    artifact: str,
    history: List[Dict[str, str]],
    prefix: str,
    thread_id: str,
) -> AsyncIterator[Dict[str, Any]]:
    """One assistant turn: route first, then stream from the bounded pool.

    Yields event dicts; the caller frames them as SSE. It never yields "done" —
    the framing wrapper owns that, so every stream ends exactly once.
    """
    if _smoke_mode():
        async for event in _smoke_events():
            yield event
        return

    try:
        # This account's own backend when it saved one, the deployment's otherwise.
        workflow = await asyncio.to_thread(get_workflow, user_id)
    except Exception as exc:
        yield {"type": "error", "detail": f"{_label()}: {exc}"}
        return

    try:
        decision = await asyncio.to_thread(router.decide, workflow.sysml_agent.llm, message)
    except Exception as exc:
        yield {"type": "error", "detail": f"{_label()}: {exc}"}
        return
    yield {
        "type": "route",
        "route": decision.route,
        "mode": decision.route,  # legacy field, same value now
        "reason": decision.reason,
        "source": decision.source,
        "format": decision.format,
    }

    queue_gate, run_gate = _turn_gates()
    if not await _admit(queue_gate, run_gate):
        yield {
            "type": "error",
            "detail": "every worker is busy right now; please retry shortly",
            "status": 503,
        }
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    future = None
    try:
        # One retrieval for both routes, so the trace row and the prompt cannot
        # disagree about what the model was shown.
        matches: Optional[List[Dict[str, str]]] = None
        if doc_ids:
            matches = _retrieved(user_id, doc_ids, message)
            yield _retrieval_event(matches)
        if decision.route == GENERAL_CHAT:
            # The chat branch already carries the conversation as messages, so the
            # transcript block would only repeat it.
            llm_messages = list(history) + [
                {
                    "role": "user",
                    "content": _compose_prompt(user_id, message, GENERAL_CHAT, matches),
                }
            ]
            future = loop.run_in_executor(
                TURN_POOL,
                _pump_chat,
                workflow.common_llm_instance,
                llm_messages,
                loop,
                queue,
                stop,
            )
        else:
            # The stored transcript rides in the prompt, so the checkpointer is
            # only a best-effort cache: another worker can serve the next turn.
            prompt = prefix + _compose_prompt(
                user_id, message, SYSML_GENERATION, matches, artifact, decision.format
            )
            future = loop.run_in_executor(
                TURN_POOL,
                _pump_pipeline,
                workflow.sysml_agent,
                prompt,
                thread_id,
                loop,
                queue,
                stop,
            )

        try:
            async with asyncio.timeout(TURN_TIMEOUT):
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), HEARTBEAT)
                    except (asyncio.TimeoutError, TimeoutError):
                        yield {"type": "ping"}
                        continue
                    if event is None:
                        break
                    yield event
        except (asyncio.TimeoutError, TimeoutError):
            stop.set()
            future.cancel()
            yield {
                "type": "error",
                "detail": f"turn timed out after {int(TURN_TIMEOUT)}s",
            }
    except asyncio.CancelledError:
        stop.set()
        if future is not None:
            future.cancel()
        raise
    finally:
        stop.set()
        # `future` is still None if composing the prompt raised: the gates must be
        # released either way, so nothing here may assume a submitted turn.
        if future is not None and not future.done():
            future.cancel()
        run_gate.release()
        queue_gate.release()


async def _sse_stream(
    events: AsyncIterator[Dict[str, Any]], user_id: str
) -> AsyncIterator[str]:
    """Frame events as SSE, remember the validated model, close with "done"."""
    async for event in events:
        if event.get("type") == "code":
            _remember_model(user_id, str(event.get("code") or ""))
        yield _sse(event)
    yield _sse({"type": "done"})


# --- Auth -------------------------------------------------------------------


def _cookie_secure() -> bool:
    return (os.environ.get("SYSML_COOKIE_SECURE") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        auth.SESSION_COOKIE,
        token,
        max_age=auth.SESSION_TTL_DAYS * 24 * 3600,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_cookie_secure(),
    )


def _client_ip(request: Request) -> str:
    """Best-effort client address for the rate limiters.

    X-Forwarded-For is only believed when SYSML_TRUST_PROXY is set, because the
    header is client-controlled: behind nginx it is the real address, but on a
    directly published port it would let one client rotate through identities and
    dodge the per-address lockout.
    """
    if (os.environ.get("SYSML_TRUST_PROXY") or "0").strip().lower() in ("1", "true", "yes"):
        forwarded = request.headers.get("x-forwarded-for") or ""
        first = forwarded.split(",")[0].strip()
        if first:
            return first[:64]
    return (request.client.host if request.client else "unknown")[:64]


def current_user(request: Request) -> Dict[str, Any]:
    """The signed-in account, or 401. Every data route depends on this."""
    user = auth.user_from_token(request.cookies.get(auth.SESSION_COOKIE))
    if user is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return user


def require_admin(user: Dict[str, Any] = Depends(current_user)) -> Dict[str, Any]:
    """Deployment-wide switches (provider, reset, approval store) are admin only."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="administrator only")
    return user


@app.post("/api/auth/register")
async def register(payload: Credentials):
    try:
        user = await asyncio.to_thread(auth.register, payload.email, payload.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message) from None
    token = await asyncio.to_thread(auth.create_session, user["id"])
    response = JSONResponse({"user": user})
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/login")
async def login(payload: Credentials, request: Request):
    key = f"{_client_ip(request)}:{payload.email.strip().lower()}"
    retry = limits.exceeded("auth", key)
    if retry is not None:
        raise HTTPException(
            status_code=429,
            detail="too many failed sign-ins; try again later",
            headers={"Retry-After": str(retry)},
        )
    user = await asyncio.to_thread(auth.authenticate, payload.email, payload.password)
    if user is None:
        # Count only failures, so a legitimate session never burns its own budget.
        limits.enforce("auth", key)
        raise HTTPException(status_code=401, detail="email or password is wrong")
    limits.reset("auth", key)
    token = await asyncio.to_thread(auth.create_session, user["id"])
    response = JSONResponse({"user": user})
    _set_session_cookie(response, token)
    return response


@app.post("/api/auth/logout")
async def logout(request: Request, user=Depends(current_user)):
    await asyncio.to_thread(auth.revoke_session, request.cookies.get(auth.SESSION_COOKIE))
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.SESSION_COOKIE, path="/")
    return response


@app.get("/api/auth/me")
async def me(user=Depends(current_user)):
    return {"user": user, "signup_open": auth.signup_open()}


# --- Health and readiness ---------------------------------------------------


def _kernel_status() -> Optional[str]:
    """TCP reachability of the visualizer sidecar (no HTTP, no workflow)."""
    url = os.environ.get("SYSML_KERNEL_SERVICE_URL", "http://sysml-kernel:8888")
    parts = urlsplit(url)
    host, port = parts.hostname, parts.port or 80
    if not host:
        return f"could not parse SYSML_KERNEL_SERVICE_URL={url!r}"
    try:
        with socket.create_connection((host, port), timeout=2.0):
            return None
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"


def _knowledge_base_status() -> Optional[str]:
    """The stores' directories must exist before a workflow tries to open them."""

    def probe() -> None:
        for path in (
            STATE_DIR,
            ROOT / "rag" / "chroma_db",
            ROOT / "rag" / "approved_solutions",
        ):
            path.mkdir(parents=True, exist_ok=True)

    return db.error_status(probe)


@app.get("/api/health")
async def health(request: Request):
    """Liveness: cheap, no workflow build.

    The counters are reported only when this worker already has a workflow
    cached — asking for them must never be what triggers the slow build.

    backend/model/temp are this account's effective settings when a session
    cookie resolves, otherwise the deployment view: the container healthcheck
    calls this anonymously. That is one session lookup, and a failure there
    must not take the probe down.
    """
    try:
        user = auth.user_from_token(request.cookies.get(auth.SESSION_COOKIE))
    except Exception:
        user = None
    env = effective_env(user["id"] if user else None)
    provider, model = active_backend(env)
    payload: Dict[str, Any] = {
        "ok": True,
        "ready": True,
        "version": os.environ.get("SYSML_VERSION") or "dev",
        "provider": provider,
        "model": model,
        "temperature": active_temperature(env),
        "base_url": env.get("SYSML_BASE_URL", ""),
        "api_key_set": bool(env.get("SYSML_API_KEY")),
        "deepagents_version": _version("deepagents"),
    }
    with _workflow_lock:
        # Whatever this worker already built, for the counters below. Reading
        # the cache must never be what triggers the slow build.
        warm = next(iter(_workflows.values()), None)
    if warm is not None:
        try:
            payload["rag_chunks"] = warm.get_rag_statistics().get("total_chunks", 0)
            payload["approved_entries"] = warm.get_approved_db_statistics().get(
                "total_entries", 0
            )
        except Exception:
            pass
    return payload


@app.get("/api/ready")
async def ready(response: Response):
    """Readiness: fail while any dependency is unusable, naming the culprit."""
    components = {
        "postgres": await asyncio.to_thread(db.postgres_status),
        "redis": await asyncio.to_thread(db.redis_status),
        "kernel": await asyncio.to_thread(_kernel_status),
        "knowledge_base": await asyncio.to_thread(_knowledge_base_status),
    }
    failed = [name for name, error in components.items() if error]
    payload: Dict[str, Any] = {
        "ready": not failed,
        "components": {
            name: ("ok" if error is None else str(error))
            for name, error in components.items()
        },
    }
    if failed:
        response.status_code = 503
        payload["failed"] = failed
    return payload


# --- Chat -------------------------------------------------------------------


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request, user=Depends(current_user)):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    limits.enforce("chat", _owner_id(user))

    thread_id = uuid.uuid4().hex
    if payload.thread_id:
        # A thread id from the client must be one of *theirs*: a foreign id is a
        # bad request rather than a turn inside someone else's conversation.
        stored = await asyncio.to_thread(
            threads.load_thread, _owner_id(user), payload.thread_id
        )
        if stored is None:
            raise HTTPException(status_code=400, detail="unknown thread_id")
        thread_id = stored["id"]

    events = _turn_events(
        user_id=_owner_id(user),
        message=message,
        doc_ids=payload.doc_ids,
        artifact=payload.artifact,
        history=_sanitize_history(payload.history),
        prefix="",
        thread_id=thread_id,
    )
    return StreamingResponse(
        _sse_stream(events, _owner_id(user)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


# --- Threads ----------------------------------------------------------------


@app.get("/api/threads")
async def list_threads(user=Depends(current_user)):
    return {"threads": await asyncio.to_thread(threads.list_threads, _owner_id(user))}


@app.post("/api/threads")
async def create_thread_route(
    payload: Optional[ThreadCreate] = None, user=Depends(current_user)
):
    title = payload.title if payload is not None else ""
    thread = await asyncio.to_thread(threads.create_thread, _owner_id(user), title)
    if thread is None:
        raise HTTPException(status_code=400, detail="could not create a thread")
    return {"thread": threads.summary(thread)}


@app.get("/api/threads/{thread_id}")
async def read_thread(thread_id: str, user=Depends(current_user)):
    thread = await asyncio.to_thread(threads.load_thread, _owner_id(user), thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"thread": thread}


@app.patch("/api/threads/{thread_id}")
async def rename_thread_route(
    thread_id: str, payload: ThreadRename, user=Depends(current_user)
):
    if not payload.title.strip():
        raise HTTPException(status_code=400, detail="title is required")
    thread = await asyncio.to_thread(
        threads.rename_thread, _owner_id(user), thread_id, payload.title
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"thread": threads.summary(thread)}


@app.delete("/api/threads/{thread_id}")
async def delete_thread_route(thread_id: str, user=Depends(current_user)):
    if not await asyncio.to_thread(threads.delete_thread, _owner_id(user), thread_id):
        raise HTTPException(status_code=404, detail="thread not found")
    return {"ok": True, "id": thread_id}


@app.post("/api/threads/{thread_id}/chat")
async def thread_chat(
    thread_id: str, payload: ThreadChatRequest, user=Depends(current_user)
):
    """One turn inside a stored thread: route, stream, then save both turns."""
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    limits.enforce("chat", _owner_id(user))
    owner = _owner_id(user)

    thread = await asyncio.to_thread(threads.load_thread, owner, thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")

    # Full stored history goes to the model; the turn is saved before the
    # stream so a dropped connection still leaves the question in the thread.
    history = threads.history_messages(thread)
    prefix = _history_block(
        [
            {"role": turn["role"], "text": turn["content"]}
            for turn in (thread.get("messages") or [])
            if str(turn.get("content", "")).strip()
        ]
    )
    await asyncio.to_thread(threads.append_message, owner, thread_id, "user", message)

    reply: List[str] = []
    model_code = ""
    last_validation: Dict[str, Any] = {}

    def save() -> Optional[Dict[str, Any]]:
        """Store the turn, creating the artifact it produced first.

        The artifact has to exist before the turn is written, because the turn
        carries its id. Called once, from the normal path or the stop path.
        """
        captured = None
        if model_code.strip():
            captured = artifacts.create_artifact(
                owner, thread_id, message, "code", model_code
            )
        # Model and verdict ride on the turn as columns: a code-only turn has no
        # prose and would otherwise be dropped by the transcript reader.
        threads.append_message(
            owner,
            thread_id,
            "assistant",
            "".join(reply).strip(),
            model=model_code,
            validation=last_validation,
            artifact_id=(captured or {}).get("id"),
        )
        return captured

    async def events() -> AsyncIterator[str]:
        nonlocal model_code, last_validation
        try:
            async for event in _turn_events(
                user_id=owner,
                message=message,
                doc_ids=payload.doc_ids,
                artifact=payload.artifact,
                history=history,
                prefix=prefix,
                thread_id=thread_id,
            ):
                kind = event.get("type")
                if kind == "token":
                    reply.append(str(event.get("text", "")))
                elif kind == "code":
                    model_code = str(event.get("code", ""))
                    _remember_model(owner, model_code)
                elif kind == "validation":
                    last_validation = {
                        key: value for key, value in event.items() if key != "type"
                    }
                yield _sse(event)
        except asyncio.CancelledError:
            # Stop button or dropped connection: keep what the turn had so far.
            await asyncio.to_thread(save)
            raise
        captured = await asyncio.to_thread(save)
        # Re-read: the first user turn is what seeds the derived title.
        final = await asyncio.to_thread(threads.load_thread, owner, thread_id) or thread
        if captured:
            # Metadata only: the strip selects the new tab, then fetches its body.
            yield _sse({"type": "artifact", "artifact": artifacts.summary(captured)})
        yield _sse({"type": "title", "title": final.get("title", "")})
        yield _sse({"type": "done"})

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)


# --- Artifacts (the canvas) -------------------------------------------------


def _kind_or_400(value: str) -> str:
    """Canvas documents are code or markdown; anything else is the client's bug."""
    kind = str(value or "").strip().lower()
    if kind not in artifacts.KINDS:
        raise HTTPException(status_code=400, detail="kind must be code or markdown")
    return kind


@app.get("/api/threads/{thread_id}/artifacts")
async def list_artifacts_route(thread_id: str, user=Depends(current_user)):
    owner = _owner_id(user)
    # Owning the thread is the precondition, so a foreign id is a 404 rather
    # than an empty list that reads as "this thread has none".
    if await asyncio.to_thread(threads.load_thread, owner, thread_id) is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"artifacts": await asyncio.to_thread(artifacts.list_artifacts, owner, thread_id)}


@app.post("/api/threads/{thread_id}/artifacts")
async def create_artifact_route(
    thread_id: str, payload: ArtifactCreate, user=Depends(current_user)
):
    owner = _owner_id(user)
    limits.enforce("canvas", owner)
    artifact = await asyncio.to_thread(
        artifacts.create_artifact,
        owner,
        thread_id,
        payload.title,
        _kind_or_400(payload.kind),
        payload.content,
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"artifact": artifact}


@app.get("/api/artifacts/{artifact_id}")
async def read_artifact(
    artifact_id: str, version: Optional[int] = None, user=Depends(current_user)
):
    """The artifact plus one version's content: current unless ?version=N."""
    artifact = await asyncio.to_thread(
        artifacts.load_artifact, _owner_id(user), artifact_id, version
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"artifact": artifact}


@app.post("/api/artifacts/{artifact_id}/versions")
async def save_artifact_version(
    artifact_id: str, payload: ArtifactVersionIn, user=Depends(current_user)
):
    owner = _owner_id(user)
    limits.enforce("canvas", owner)
    artifact = await asyncio.to_thread(
        artifacts.add_version,
        owner,
        artifact_id,
        _kind_or_400(payload.kind),
        payload.content,
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"artifact": artifact}


@app.post("/api/artifacts/{artifact_id}/restore")
async def restore_artifact(
    artifact_id: str, payload: ArtifactRestore, user=Depends(current_user)
):
    """Append an older version's content as a new version: history is kept."""
    owner = _owner_id(user)
    limits.enforce("canvas", owner)
    source = await asyncio.to_thread(
        artifacts.load_artifact, owner, artifact_id, payload.version
    )
    if source is None:
        raise HTTPException(status_code=404, detail="artifact version not found")
    artifact = await asyncio.to_thread(
        artifacts.add_version,
        owner,
        artifact_id,
        source["kind"],
        source["content"],
        "restore",
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"artifact": artifact}


@app.delete("/api/artifacts/{artifact_id}")
async def delete_artifact_route(artifact_id: str, user=Depends(current_user)):
    owner = _owner_id(user)
    limits.enforce("canvas", owner)
    if not await asyncio.to_thread(artifacts.delete_artifact, owner, artifact_id):
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"ok": True, "id": artifact_id}


# --- Documents --------------------------------------------------------------


@app.post("/api/documents", response_model=Dict[str, List[DocumentInfo]])
async def upload_documents(
    files: List[UploadFile] = File(...), user=Depends(current_user)
):
    """Read engineering documents (txt/md/sysml/pdf/docx) into this account's store."""
    limits.enforce("upload", _owner_id(user))
    added: List[Dict[str, Any]] = []
    for upload in files:
        name = upload.filename or "document"
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix not in documents.SUFFIXES:
            raise HTTPException(status_code=400, detail=f"unsupported file type: {name}")
        data = await upload.read()
        if not data:
            continue
        if len(data) > 25_000_000:
            raise HTTPException(status_code=400, detail=f"file too large: {name}")
        try:
            document = await asyncio.to_thread(
                documents.add_document, _owner_id(user), name, data
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not read {name}: {exc}")
        if document is None:
            raise HTTPException(status_code=400, detail=f"could not store {name}")
        added.append(
            {
                "id": document.id,
                "name": document.name,
                "chars": document.chars,
                "chunks": len(document.chunks),
            }
        )
    if not added:
        raise HTTPException(status_code=400, detail="no readable content")
    return {"documents": added}


@app.get("/api/documents", response_model=Dict[str, List[DocumentInfo]])
async def list_documents(user=Depends(current_user)):
    stored = await asyncio.to_thread(documents.list_documents, _owner_id(user))
    return {
        "documents": [
            {"id": d.id, "name": d.name, "chars": d.chars, "chunks": len(d.chunks)}
            for d in stored
        ]
    }


def _upload_name(raw: str) -> str:
    """Reduce a client-supplied filename to a safe leaf name under the uploads dir."""
    name = Path(raw.replace("\\", "/")).name.strip()
    if name in ("", ".", "..") or chr(0) in name:
        raise ValueError(f"invalid file name: {raw!r}")
    return name


@app.post("/api/workspace/files", response_model=Dict[str, Any])
async def upload_workspace_files(
    files: List[UploadFile] = File(...), user=Depends(current_user)
):
    """Put uploaded files in this account's sandbox directory as agent input."""
    limits.enforce("upload", _owner_id(user))
    workflow = await asyncio.to_thread(get_workflow, _owner_id(user))
    sandbox = workflow.sysml_agent.sandbox
    uploads = workspace_paths(_owner_id(user))["uploads"]

    payloads: List[tuple] = []
    sizes: Dict[str, int] = {}
    errors: List[Dict[str, str]] = []
    for upload in files:
        raw = upload.filename or "upload"
        try:
            name = _upload_name(raw)
        except ValueError as exc:
            errors.append({"name": raw, "detail": str(exc)})
            continue
        data = await upload.read()
        if not data:
            errors.append({"name": name, "detail": "empty file"})
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(
                {"name": name, "detail": f"file too large (limit {MAX_UPLOAD_BYTES} bytes)"}
            )
            continue
        path = str(uploads / name)
        payloads.append((path, data))
        sizes[path] = len(data)

    stored: List[Dict[str, Any]] = []
    if payloads:
        responses = await asyncio.to_thread(sandbox.upload_files, payloads)
        for response in responses:
            if response.error:
                errors.append({"name": Path(response.path).name, "detail": response.error})
            else:
                stored.append(
                    {
                        "name": Path(response.path).name,
                        "path": response.path,
                        "bytes": sizes.get(response.path, 0),
                    }
                )
    if not stored:
        detail = "; ".join(f"{e['name']}: {e['detail']}" for e in errors) or "no files received"
        raise HTTPException(status_code=400, detail=detail)
    return {"files": stored, "errors": errors}


@app.get("/api/workspace/files", response_model=Dict[str, Any])
async def list_workspace_files(user=Depends(current_user)):
    """List this account's input files, as the agent's sandbox sees them."""
    workflow = await asyncio.to_thread(get_workflow, _owner_id(user))
    uploads = str(workspace_paths(_owner_id(user))["uploads"])
    result = await asyncio.to_thread(workflow.sysml_agent.sandbox.ls, uploads)
    files = [
        {"name": Path(entry["path"]).name, "path": entry["path"]}
        for entry in (result.entries or [])
        if entry.get("path") and not entry.get("is_dir")
    ]
    return {"files": files, "error": getattr(result, "error", None)}


# --- Rendering --------------------------------------------------------------


@app.post("/api/render")
async def render_model(request: RenderRequest, user=Depends(current_user)):
    """Render SysML v2 to an SVG/PNG data URL with the kernel visualizer.

    Send "code" to render an arbitrary model, or omit it to render this account's
    last validated model.
    """
    owner = _owner_id(user)
    code = (request.code or "").strip()
    revision = 0
    if not code:
        state = await asyncio.to_thread(model_state, owner)
        code = state["code"].strip()
        revision = state["revision"]
    if not code:
        raise HTTPException(
            status_code=409, detail="no model yet: send code or generate one first"
        )
    element_name = request.element_name.strip()
    key = (
        owner,
        hashlib.sha256(code.encode("utf-8")).hexdigest(),
        element_name,
        request.orientation,
        request.compartments,
    )
    cached = _render_cached(key)
    if cached is not None:
        return {**cached, "cached": True}

    reply = dict(
        await asyncio.to_thread(
            _kernel_render,
            code,
            element_name,
            request.orientation,
            request.compartments,
        )
    )
    reply.update(
        {
            "element_name": element_name,
            "orientation": request.orientation,
            "compartments": request.compartments,
            "revision": revision,
            "cached": False,
        }
    )
    if reply.get("success"):
        _render_store(key, reply)
    return reply


@app.get("/api/render/state")
async def render_state(user=Depends(current_user)):
    """Cheap change signal for a diagram panel; never triggers a render."""
    state = await asyncio.to_thread(model_state, _owner_id(user))
    return {
        "revision": state["revision"],
        "has_model": bool(state["code"].strip()),
    }


@app.get("/api/render/latest")
async def render_latest(
    orientation: str = "TB",
    compartments: str = "default",
    element_name: str = "",
    user=Depends(current_user),
):
    """Render this account's last validated model (for a sidebar panel)."""
    state = await asyncio.to_thread(model_state, _owner_id(user))
    if not state["code"].strip():
        raise HTTPException(status_code=409, detail="no validated model yet")
    return await render_model(
        RenderRequest(
            code=state["code"],
            element_name=element_name,
            orientation=orientation,
            compartments=compartments,
        ),
        user,
    )


# --- Analysis ---------------------------------------------------------------


@app.post("/api/validate")
async def validate(request: ValidateRequest, user=Depends(current_user)):
    """ANTLR verdict for a pasted model - no LLM involved."""
    code = request.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="code is required")

    def run() -> Dict[str, Any]:
        from webapp.validation import format_validation_result, validate_sysml_code

        result = validate_sysml_code(code)
        return {**result, "detail": format_validation_result(result)}

    return await asyncio.to_thread(run)


@app.post("/api/model/index")
async def model_index(request: ModelRequest, user=Depends(current_user)):
    """Elements, edges, issues, review and traceability for one model.

    The client sends the code, so a panel opened on an old turn indexes that
    turn's model and nothing else. The ANTLR walk is CPU-bound, hence the thread.
    """
    code = (request.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="model is required")

    def run() -> Dict[str, Any]:
        from webapp.model_analysis import report

        return report(code)

    return await asyncio.to_thread(run)


@app.post("/api/model/impact")
async def model_impact(request: ModelImpactRequest, user=Depends(current_user)):
    """What would notice a change to one element: reverse reachability."""
    code = (request.code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="model is required")
    element_id = (request.element_id or "").strip()
    if not element_id:
        raise HTTPException(status_code=400, detail="element_id is required")

    def run() -> Dict[str, Any]:
        from webapp.model_analysis import impact
        from webapp.model_index import build_index

        try:
            return impact(build_index(code), element_id)
        except KeyError:
            raise HTTPException(status_code=400, detail="unknown element") from None

    return await asyncio.to_thread(run)


@app.get("/api/tools")
async def tools(user=Depends(current_user)):
    """Tool manifest: the client keys its workspace UI off these names."""
    return {"tools": TOOL_MANIFEST}


@app.post("/api/approve")
async def approve(request: ApprovalRequest, user=Depends(require_admin)):
    """Store a validated model in the shared approved-solutions store."""
    code = request.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="code is required")

    def store() -> Dict[str, Any]:
        from states.ValidationState import ValidationStatus
        from states.WorkflowState import WorkflowState
        from webapp.validation import format_validation_result, validate_sysml_code

        workflow = get_workflow(_owner_id(user))
        result = validate_sysml_code(code)
        verdict = format_validation_result(result)
        if result.get("status") != "valid":
            return {"stored": False, "validation": verdict}

        task = request.task.strip() or "web approval"
        state = WorkflowState(
            original_query=task,
            processed_query=task,
            code=code,
            human_feedback=request.feedback,
            approval_status="approved",
            is_valid=ValidationStatus.VALID,
        )
        # One writer at a time across workers: Chroma's SQLite store is shared.
        with locks.flock():
            workflow._update_knowledge_base(state)
        stats = workflow.get_approved_db_statistics()
        return {
            "stored": True,
            "validation": verdict,
            "approved_entries": stats.get("total_entries", 0),
        }

    try:
        return await asyncio.to_thread(store)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"{_label()}: {exc}")


# --- Per-account provider configuration -------------------------------------


class ProviderConfigRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    effort: Optional[str] = None


def _deployment_view() -> Dict[str, Any]:
    """The deployment's own backend, as the per-account dialog opens on it."""
    env = effective_env()
    provider, model = active_backend(env)
    return {
        "provider": provider,
        "model": model,
        "temperature": active_temperature(env),
        "base_url": env.get("SYSML_BASE_URL", ""),
        "effort": env.get("SYSML_EFFORT", ""),
        "api_key_set": bool(env.get("SYSML_API_KEY")),
    }


@app.get("/api/provider-config")
async def read_provider_config(user=Depends(current_user)):
    """This account's own backend, or the deployment default it falls back to."""
    return providers.view(_owner_id(user), _deployment_view())


@app.put("/api/provider-config")
async def save_provider_config(
    payload: ProviderConfigRequest, user=Depends(current_user)
):
    """Point this account's own turns at its own backend.

    The account comes from the session, never from the body, so one account can
    only ever write its own row. A blank api_key keeps the stored one (the key
    itself never goes back to the browser), and a base url is validated by
    providers.guard_base_url before it is stored, because this server will fetch
    it on every one of this account's turns.
    """
    provider = (payload.provider or "").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise HTTPException(status_code=400, detail=f"unknown provider: {provider}")
    model = (payload.model or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if payload.temperature is not None and not 0 <= payload.temperature <= 2:
        raise HTTPException(
            status_code=400, detail="temperature must be between 0 and 2"
        )
    try:
        base_url = providers.guard_base_url(payload.base_url)
        await asyncio.to_thread(
            providers.save,
            _owner_id(user),
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "api_key": payload.api_key or "",
                "temperature": payload.temperature,
                "effort": (payload.effort or "").strip(),
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return providers.view(_owner_id(user), _deployment_view())


@app.delete("/api/provider-config")
async def clear_provider_config(user=Depends(current_user)):
    """Drop this account's own backend: the deployment default applies again."""
    await asyncio.to_thread(providers.clear, _owner_id(user))
    return providers.view(_owner_id(user), _deployment_view())


# --- Configuration ----------------------------------------------------------


class ConfigRequest(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    effort: Optional[str] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    # Apply these settings and keep them under this name ("save as").
    name: Optional[str] = None


@app.post("/api/config")
async def configure(request: ConfigRequest, user=Depends(require_admin)):
    """Apply backend settings for the whole deployment; the workflow rebuilds."""
    payload: Dict[str, str] = {}

    if request.provider is not None:
        provider = request.provider.strip().lower()
        if provider not in DEFAULT_MODELS:
            raise HTTPException(status_code=400, detail=f"unknown provider: {provider}")
        payload["SYSML_PROVIDER"] = provider
    if request.temperature is not None:
        if not 0 <= request.temperature <= 2:
            raise HTTPException(
                status_code=400, detail="temperature must be between 0 and 2"
            )
        payload["SYSML_TEMPERATURE"] = str(request.temperature)
    if request.max_tokens is not None:
        if request.max_tokens < 1:
            raise HTTPException(status_code=400, detail="max_tokens must be positive")
        payload["SYSML_MAX_TOKENS"] = str(int(request.max_tokens))
    if request.timeout is not None:
        if request.timeout < 1:
            raise HTTPException(status_code=400, detail="timeout must be positive")
        payload["SYSML_TIMEOUT"] = str(int(request.timeout))
    if request.model and request.model.strip():
        payload["SYSML_MODEL"] = request.model.strip()
    if request.base_url is not None:
        payload["SYSML_BASE_URL"] = request.base_url.strip()
    if request.api_key:
        payload["SYSML_API_KEY"] = request.api_key.strip()
    if request.effort:
        payload["SYSML_EFFORT"] = request.effort.strip()

    state = settings_snapshot(force=True)
    overrides = dict(state["overrides"])
    overrides.update(payload)

    # A named save stores exactly what was submitted, so one configuration can
    # never inherit a field the user set for another. The api key is the single
    # exception: the browser never receives it back, so it is carried over only
    # when re-saving the configuration that is live right now.
    profiles = dict(state["profiles"]["profiles"])
    saved_as = (request.name or "").strip()
    if saved_as:
        saved = dict(payload)
        if "SYSML_API_KEY" not in saved and state["profiles"]["active"] == saved_as:
            if "SYSML_API_KEY" in overrides:
                saved["SYSML_API_KEY"] = overrides["SYSML_API_KEY"]
        profiles[saved_as] = saved

    _write_settings(
        {
            "overrides": overrides,
            "profiles": {"active": saved_as or None, "profiles": profiles},
        }
    )
    drop_workflow()

    env = effective_env()
    provider, model = active_backend(env)
    return {
        "ok": True,
        "provider": provider,
        "model": model,
        "temperature": active_temperature(env),
        "base_url": env.get("SYSML_BASE_URL", ""),
        "api_key_set": bool(env.get("SYSML_API_KEY")),
        "persisted": True,
        "active": saved_as or None,
    }


@app.get("/api/config")
async def read_config(user=Depends(require_admin)):
    """Current provider settings, the saved configurations, and the active one."""
    env = effective_env()
    provider, model = active_backend(env)
    store = settings_snapshot()["profiles"]
    return {
        "provider": provider,
        "model": model,
        "temperature": active_temperature(env),
        "base_url": env.get("SYSML_BASE_URL", ""),
        "effort": env.get("SYSML_EFFORT", ""),
        "max_tokens": env.get("SYSML_MAX_TOKENS"),
        "timeout": env.get("SYSML_TIMEOUT"),
        "api_key_set": bool(env.get("SYSML_API_KEY")),
        "persisted": True,
        "active": store["active"],
        "profiles": [
            _profile_view(name, values)
            for name, values in sorted(store["profiles"].items())
        ],
    }


@app.post("/api/config/profiles/{name}/apply")
async def apply_profile(name: str, user=Depends(require_admin)):
    """Switch to a saved configuration.

    Wholesale replace, not a merge: otherwise a key set only in the previous
    configuration (an api key, a base url) would leak into this one.
    """
    state = settings_snapshot(force=True)
    values = state["profiles"]["profiles"].get(name)
    if values is None:
        raise HTTPException(status_code=404, detail=f"unknown configuration: {name}")

    _write_settings(
        {
            "overrides": dict(values),
            "profiles": {"active": name, "profiles": state["profiles"]["profiles"]},
        }
    )
    drop_workflow()

    env = effective_env()
    provider, model = active_backend(env)
    return {
        "ok": True,
        "active": name,
        "provider": provider,
        "model": model,
        "temperature": active_temperature(env),
        "base_url": env.get("SYSML_BASE_URL", ""),
        "api_key_set": bool(env.get("SYSML_API_KEY")),
        "persisted": True,
    }


@app.delete("/api/config/profiles/{name}")
async def delete_profile(name: str, user=Depends(require_admin)):
    """Forget a saved configuration; the current settings stay applied."""
    state = settings_snapshot(force=True)
    profiles = dict(state["profiles"]["profiles"])
    if name not in profiles:
        raise HTTPException(status_code=404, detail=f"unknown configuration: {name}")
    del profiles[name]
    active = None if state["profiles"]["active"] == name else state["profiles"]["active"]
    _write_settings(
        {
            "overrides": state["overrides"],
            "profiles": {"active": active, "profiles": profiles},
        }
    )
    return {"ok": True, "active": active}


@app.post("/api/reset")
async def reset(user=Depends(require_admin)):
    """Drop the cached workflow, the saved provider settings and this admin's uploads."""
    _write_settings(
        {
            "overrides": {},
            "profiles": {"active": None, "profiles": {}},
        }
    )
    drop_workflow()
    await asyncio.to_thread(documents.reset_documents, _owner_id(user))
    provider, model = active_backend()
    return {"ok": True, "provider": provider, "model": model}


# --- App --------------------------------------------------------------------

_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_MULTIPART_ROUTES = {"/api/documents", "/api/workspace/files"}


def _origin_allowed(origin: str, host: str) -> bool:
    """Same host (any scheme), or explicitly listed in SYSML_ALLOWED_ORIGINS."""
    allowed = {
        item.strip().rstrip("/")
        for item in (os.environ.get("SYSML_ALLOWED_ORIGINS") or "").split(",")
        if item.strip()
    }
    if origin.rstrip("/") in allowed:
        return True
    try:
        netloc = urlsplit(origin).netloc.lower()
    except ValueError:
        return False
    return bool(host) and netloc == host.lower()


async def _reject(send, status: int, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class OriginGuardMiddleware:
    """Reject cross-site writes and keep mutating routes JSON.

    A cookie-authenticated API is otherwise writable from any page the user
    visits: SameSite=Lax blocks most of it, and this closes the rest by checking
    the Origin header whenever the browser sends one.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method", "GET")).upper()
        if method in _UNSAFE_METHODS:
            headers = {
                name.decode("latin-1").lower(): value.decode("latin-1")
                for name, value in scope.get("headers") or []
            }
            origin = headers.get("origin", "")
            if origin and not _origin_allowed(origin, headers.get("host", "")):
                await _reject(send, 403, "cross-origin request rejected")
                return
            path = scope.get("path", "")
            content_type = headers.get("content-type", "").lower()
            if (
                path.startswith("/api/")
                and path not in _MULTIPART_ROUTES
                and content_type
                and "application/json" not in content_type
            ):
                await _reject(send, 415, "content-type must be application/json")
                return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logs.configure_logging()
    try:
        applied = await asyncio.to_thread(db.apply_migrations)
        if applied:
            logger.info("applied migrations: %s", ", ".join(applied))
    except Exception as exc:
        # Keep serving: /api/ready reports the database, which is what a load
        # balancer needs to see, and an operator gets the log line.
        logger.error("migrations failed: %s", exc)
    for path in (STATE_DIR, ROOT / "rag" / "chroma_db", ROOT / "rag" / "approved_solutions"):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    yield
    TURN_POOL.shutdown(wait=False, cancel_futures=True)
    db.close_clients()


# The app object was created before the routes; this is where its startup and
# shutdown hook lands (migrations, log format, pool teardown).
app.router.lifespan_context = lifespan
# Order matters: the last added is outermost, so the access log sees the 403s.
app.add_middleware(OriginGuardMiddleware)
app.add_middleware(logs.AccessLogMiddleware)


@app.exception_handler(limits.RateLimited)
async def rate_limited_handler(request: Request, exc: limits.RateLimited):
    return JSONResponse(
        status_code=429,
        content={"detail": str(exc), "retry_after": exc.retry_after},
        headers={"Retry-After": str(exc.retry_after)},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    # The middleware has already unwound by the time this runs, so the scope
    # carries the id; the contextvar is the fallback for a non-HTTP caller.
    request_id = getattr(request.state, "request_id", None) or logs.request_id_var.get()
    logger.error(
        "unhandled error on %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    # The id is what support needs; the traceback stays in the log. The header is
    # set here because this response is produced outside the access-log
    # middleware (Starlette's error handler wraps it), so it would not be added.
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "request_id": request_id},
        headers={"X-Request-Id": request_id},
    )


@app.get("/")
async def index():
    """One chat surface: the built React app."""
    if (UI_DIR / "index.html").is_file():
        return FileResponse(UI_DIR / "index.html")
    # A fresh checkout has no bundle; say so instead of serving a blank page.
    raise HTTPException(
        status_code=503, detail="UI not built: run npm run build in frontend"
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
