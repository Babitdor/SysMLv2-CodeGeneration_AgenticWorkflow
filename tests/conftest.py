"""Test harness: a migrated database, an ASGI client and real accounts.

Everything goes through httpx's ASGI transport instead of calling route functions
directly, because the contract under test *is* the HTTP one: cookies and 401s,
404s on someone else's ids, SSE framing, Retry-After on a rate limit.

The environment is pinned before ``webapp`` is imported: both clients are built
from DATABASE_URL/REDIS_URL on first use, and the state/workspace roots are
resolved at import time, so a test pointing them somewhere unwritable would fail
in the wrong place.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from types import SimpleNamespace

import pytest

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://sysml:sysml@127.0.0.1:55432/sysml"
)
TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://127.0.0.1:56379/15")

os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["REDIS_URL"] = TEST_REDIS_URL
os.environ["SYSML_STATE_DIR"] = os.environ.get(
    "SYSML_STATE_DIR", tempfile.mkdtemp(prefix="sysml-test-state-")
)
os.environ["SYSML_WORKSPACE_ROOT"] = os.environ.get(
    "SYSML_WORKSPACE_ROOT", tempfile.mkdtemp(prefix="sysml-test-ws-")
)
os.environ.setdefault("SYSML_COOKIE_SECURE", "0")
os.environ.setdefault("SYSML_ALLOW_SIGNUP", "1")
# Pinned, not defaulted: something in the import graph calls load_dotenv(), so a
# working tree with SYSML_SMOKE=1 in .env would otherwise have every turn answer
# with the canned smoke stream instead of the stub under test.
os.environ["SYSML_SMOKE"] = "0"

import httpx  # noqa: E402

from webapp import db  # noqa: E402

PASSWORD = "correct-horse-battery"
TABLES = (
    "users, threads, thread_messages, documents, app_settings, sessions, "
    "artifacts, artifact_versions"
)


@pytest.fixture(scope="session", autouse=True)
def migrated():
    """One migration run for the whole session; per-test truncation below."""
    db.apply_migrations()
    yield
    # The pool's worker threads are not daemons: without this the interpreter
    # waits ~5s per thread at exit and prints a "couldn't stop thread" hint.
    db.close_clients()


@pytest.fixture(autouse=True)
def fresh_state():
    """Empty tenants and empty counters before every test.

    The truncation is what lets a fixture register the first account and rely on
    it being the administrator.
    """
    with db.connection() as conn:
        conn.execute(f"truncate {TABLES} restart identity cascade")
    try:
        db.redis_client().flushdb()
    except Exception:
        pytest.fail("Redis is unreachable: start the compose stack (postgres + redis)")
    from webapp import server

    server.drop_workflow()
    server.settings_snapshot(force=True)
    yield


def _transport():
    from webapp import server

    return httpx.ASGITransport(app=server.app)


@pytest.fixture
async def client():
    """An unauthenticated client: probe routes, log in, register."""
    async with httpx.AsyncClient(transport=_transport(), base_url="http://testserver") as http:
        yield http


@pytest.fixture
async def make_user():
    """Factory for a signed-in account with its own cookie jar."""
    created = []

    async def factory(password: str = PASSWORD, email: str | None = None):
        address = email or f"{uuid.uuid4().hex[:10]}@example.test"
        http = httpx.AsyncClient(transport=_transport(), base_url="http://testserver")
        created.append(http)
        response = await http.post(
            "/api/auth/register", json={"email": address, "password": password}
        )
        assert response.status_code == 200, response.text
        return SimpleNamespace(
            client=http,
            email=address,
            password=password,
            user=response.json()["user"],
            id=response.json()["user"]["id"],
        )

    yield factory
    for http in created:
        await http.aclose()


@pytest.fixture
async def user_a(make_user):
    """The first account in an empty database: the administrator."""
    return await make_user()


@pytest.fixture
async def user_b(make_user, user_a):
    """A second, ordinary account."""
    return await make_user()


from support import event_types, sse_events  # noqa: E402,F401  (re-exported)


class StubBackend:
    """A fake provider: no network, no model, and instrumented for concurrency.

    Tests set ``route``, ``reply``, ``code`` and ``delay``; ``peak`` records the
    high-water mark of simultaneous turns, which is what proves the pool bound.
    """

    def __init__(self) -> None:
        self.route = "chat"
        self.reply = "stub reply"
        self.code = ""
        self.validation: dict = {}
        self.delay = 0.0
        self.hang = 0.0
        self.prompts: list = []
        self.peak = 0
        self.in_flight = 0
        self._guard = threading.Lock()

    def _enter(self) -> None:
        with self._guard:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)

    def _leave(self) -> None:
        with self._guard:
            self.in_flight -= 1

    def stream(self, messages):
        """One chat turn: words, one at a time."""
        self.prompts.append(messages)
        self._enter()
        try:
            time.sleep(self.delay)
            for word in self.reply.split():
                yield SimpleNamespace(text=word + " ")
        finally:
            self._leave()

    def pipeline(self, agent, prompt, thread_id):
        """One generation turn: prose, then the model, then the verdict."""
        self.prompts.append(prompt)
        self._enter()
        try:
            time.sleep(self.delay)
            if self.hang:
                time.sleep(self.hang)
            for word in self.reply.split():
                yield {"type": "token", "text": word + " "}
            if self.code:
                yield {"type": "code", "code": self.code, "attempt": 1, "final": True}
            if self.validation:
                yield {"type": "validation", **self.validation}
        finally:
            self._leave()


@pytest.fixture
def stub(monkeypatch):
    """Replace the provider, the pipeline and the router with a controllable stub."""
    from webapp import router, server

    backend = StubBackend()
    workflow = SimpleNamespace(
        sysml_agent=SimpleNamespace(llm=object(), sandbox=SimpleNamespace()),
        common_llm_instance=backend,
        temperature=0.15,
        get_rag_statistics=lambda: {"total_chunks": 0},
        get_approved_db_statistics=lambda: {"total_entries": 0},
        cleanup=lambda: None,
    )
    monkeypatch.setattr(server, "get_workflow", lambda user_id=None: workflow)
    monkeypatch.setattr(
        server,
        "run_sysml_pipeline",
        lambda agent, prompt, thread_id: backend.pipeline(agent, prompt, thread_id),
    )
    monkeypatch.setattr(
        router,
        "decide",
        lambda llm, text: SimpleNamespace(
            route=router.GENERAL_CHAT if backend.route == "chat" else router.SYSML_GENERATION,
            reason="stub",
            source="stub",
            format="v2",
        ),
    )
    return backend


@pytest.fixture
def workspace(tmp_path):
    """Point the per-user workspace at a temp dir for this test."""
    from webapp import server

    original = server.WORKSPACE_ROOT
    server.WORKSPACE_ROOT = tmp_path
    yield tmp_path
    server.WORKSPACE_ROOT = original
