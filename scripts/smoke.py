"""End-to-end smoke test against a running deployment.

    python scripts/smoke.py --base-url https://localhost -k

Registers two throwaway accounts, drives a turn over SSE and then proves the
second account cannot see any of the first's data. Exits non-zero on the first
failure, so it can gate a deploy or a compose run.

Start the app with SYSML_SMOKE=1 for canned turns (no provider call); without it
the same checks run but depend on a reachable model backend.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from typing import Any, Dict, List

import httpx

PASSWORD = "smoke-test-password"


class Failure(Exception):
    """A check that did not hold; the message says which."""


def check(condition: Any, message: str) -> None:
    if not condition:
        raise Failure(message)


def sse_events(text: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


def make_docx(paragraph: str) -> bytes:
    """A minimal Word file: enough to prove .docx extraction works in the image."""
    import io
    import zipfile

    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body>'
        f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


def register(http: httpx.Client, label: str) -> Dict[str, Any]:
    email = f"smoke-{label}-{secrets.token_hex(4)}@example.test"
    response = http.post(
        "/api/auth/register", json={"email": email, "password": PASSWORD}
    )
    check(
        response.status_code == 200,
        f"register {label}: HTTP {response.status_code} {response.text[:200]}",
    )
    user = response.json()["user"]
    check(user["email"] == email, f"register {label}: wrong email echoed")
    return user


def run(
    base_url: str, verify: bool, timeout: float, approve: bool, expect_fresh: bool = False
) -> None:
    with httpx.Client(
        base_url=base_url, verify=verify, timeout=timeout, follow_redirects=True
    ) as admin, httpx.Client(
        base_url=base_url, verify=verify, timeout=timeout, follow_redirects=True
    ) as other:
        ready = admin.get("/api/ready")
        check(ready.status_code == 200, f"/api/ready: HTTP {ready.status_code} {ready.text[:300]}")
        check(ready.json()["ready"] is True, "/api/ready: not ready")

        health = admin.get("/api/health").json()
        check(health.get("ok") is True, f"/api/health: {health}")
        print(f"  health ok: {health['provider']}/{health['model']} v{health['version']}")

        owner = register(admin, "owner")
        guest = register(other, "guest")
        # Only the very first account on a fresh database is promoted, so the
        # role assertion only holds on a new install (CI passes --expect-fresh).
        # Against a database that already has accounts the bootstrap is claimed
        # and failing here would blame the product for a test-precondition miss.
        if expect_fresh:
            check(owner["role"] == "admin", "the first account should be the administrator")
        elif owner["role"] != "admin":
            print("  admin bootstrap already claimed on this instance, role check skipped")
        check(guest["role"] == "user", "the second account should be an ordinary user")
        print(f"  accounts ok: {owner['email']} ({owner['role']}), {guest['email']}")
        # /api/approve is administrator-only; on an instance whose bootstrap is
        # already claimed this smoke account is an ordinary user.
        is_admin = owner["role"] == "admin"

        check(other.get("/api/threads").json()["threads"] == [], "the guest started with threads")

        created = admin.post("/api/threads", json={"title": "smoke"})
        check(created.status_code == 200, f"create thread: {created.status_code}")
        thread_id = created.json()["thread"]["id"]

        streamed = admin.post(
            f"/api/threads/{thread_id}/chat", json={"message": "smoke test please"}
        )
        check(streamed.status_code == 200, f"chat: HTTP {streamed.status_code}")
        events = sse_events(streamed.text)
        kinds = [event["type"] for event in events]
        check("route" in kinds, f"no route event in {kinds}")
        check("token" in kinds, f"no token events in {kinds}")
        check(kinds[-1] == "done", f"stream did not end with done: {kinds[-3:]}")
        print(f"  turn ok: {len(kinds)} events")

        stored = admin.get(f"/api/threads/{thread_id}")
        check(stored.status_code == 200, f"read back: {stored.status_code}")
        messages = stored.json()["thread"]["messages"]
        check([m["role"] for m in messages] == ["user", "assistant"], f"turns: {messages}")
        print(f"  transcript ok: {len(messages)} turns")

        # --- isolation ---------------------------------------------------
        check(other.get(f"/api/threads/{thread_id}").status_code == 404, "guest could read the thread")
        check(
            other.patch(f"/api/threads/{thread_id}", json={"title": "x"}).status_code == 404,
            "guest could rename the thread",
        )
        check(
            other.delete(f"/api/threads/{thread_id}").status_code == 404,
            "guest could delete the thread",
        )
        check(
            other.post("/api/chat", json={"message": "hi", "thread_id": thread_id}).status_code
            == 400,
            "guest could drive a turn in the thread",
        )
        check(other.get("/api/threads").json()["threads"] == [], "guest sees the owner's threads")
        print("  isolation ok: foreign thread is 404/400 everywhere")

        uploaded = admin.post(
            "/api/documents",
            files=[
                (
                    "files",
                    (
                        "smoke.docx",
                        make_docx("Traglast 50 kg, Hub 12 m"),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ),
                )
            ],
        )
        check(uploaded.status_code == 200, f"upload: {uploaded.status_code} {uploaded.text[:200]}")
        stored_doc = uploaded.json()["documents"][0]
        check(stored_doc["chars"] > 0, f"the .docx extracted no text: {stored_doc}")
        check(stored_doc["chunks"] >= 1, f"the .docx was not chunked: {stored_doc}")
        check(len(admin.get("/api/documents").json()["documents"]) == 1, "owner lost the document")
        check(
            other.get("/api/documents").json()["documents"] == [],
            "guest sees the owner's documents",
        )
        print("  documents ok: owner 1, guest 0")

        owner_state = admin.get("/api/render/state").json()
        guest_state = other.get("/api/render/state").json()
        check(guest_state == {"revision": 0, "has_model": False}, f"guest state: {guest_state}")
        check(guest_state != owner_state, "the last model is shared between accounts")
        check(
            other.get("/api/render/latest").status_code == 409,
            "guest rendered the owner's last model",
        )
        print("  last model ok: owner has one, guest has none")

        # --- the approval store, across replicas -------------------------
        if approve and not is_admin:
            print("  approvals skipped: the smoke account is not an administrator")
        if approve and is_admin:
            posted = admin.post(
                "/api/approve",
                json={"code": "package Smoke {\n    part def Ping;\n}", "task": "smoke"},
            )
            check(posted.status_code == 200, f"approve: {posted.status_code} {posted.text[:200]}")
            stored_now = posted.json()
            check(stored_now.get("stored") is True, f"approve refused: {stored_now}")

            # A fresh turn (in a pool with several workers this may land on
            # another process): the canned reply reports what that worker can
            # retrieve from the shared approved store.
            probe = sse_events(
                admin.post("/api/chat", json={"message": "smoke probe"}).text
            )
            found = [
                part
                for event in probe
                if event["type"] == "token"
                for part in str(event["text"]).split()
                if part.startswith("approved=")
            ]
            check(found, f"no approved-store probe in {probe}")
            hits = int(found[0].split("=", 1)[1])
            check(hits >= 1, f"the approved model is not retrievable ({hits})")
            print(f"  approvals ok: stored and retrievable ({hits} hits)")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "-k", "--insecure", action="store_true", help="accept a self-signed certificate"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--expect-fresh",
        action="store_true",
        help="assert the first account becomes admin (only true on a new database)",
    )
    parser.add_argument(
        "--no-approve",
        action="store_false",
        dest="approve",
        help="skip the approval-store check (it builds the workflow once per worker)",
    )
    args = parser.parse_args(argv)

    print(f"smoke: {args.base_url}")
    try:
        run(args.base_url, not args.insecure, args.timeout, args.approve, args.expect_fresh)
    except Failure as exc:
        print(f"smoke: FAILED — {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"smoke: FAILED — transport error: {exc}", file=sys.stderr)
        return 1
    print("smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
