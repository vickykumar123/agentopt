"""Attach-mode request interceptors for real frameworks.

register_server patches the ASGI (Starlette/FastAPI) and WSGI (Flask) request
paths so a trial overlay + run id sent as headers scope that one request. Each
framework's in-process test client exercises the patched call path; the LLM
call uses a mock transport, so there is no network.

Framework types are imported at module scope on purpose: with
``from __future__ import annotations`` in effect, FastAPI resolves a handler's
annotations via the module globals, so ``Request`` must live there.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from flask import Flask, jsonify, request
from openai import AsyncOpenAI, OpenAI

from agentopt.models.trace import LlmCall, parse_trace_line
from agentopt.paths import Workspace
from agentopt.wiretap.overlay import Overlay, OverlayRule

SYSTEM = "You are a terse fitness coach."

FRENCH_OVERLAY = Overlay(
    rules=[
        OverlayRule(
            entry_id="e",
            kind="literal",
            match_exact=SYSTEM,
            new_template=SYSTEM + " Reply in French.",
        )
    ]
).to_env()


def _fake_llm(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    system = next(m["content"] for m in body["messages"] if m["role"] == "system")
    reply = "bonjour" if "Reply in French" in system else "hi there"
    return httpx.Response(
        200,
        json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": reply},
                 "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        },
    )


@pytest.fixture
def attach_server(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Register the wiretap in server mode for one test, then fully unwind it."""
    import agentopt.wiretap.context as context
    import agentopt.wiretap.session as session
    from agentopt.wiretap.instrument import manager

    session._server_registered = False
    session.register_server(project_root=tmp_path)
    try:
        yield tmp_path
    finally:
        context._global_run = None
        context._current_run.set(None)
        manager.release()
        session._server_registered = False


def _stamped_calls(tmp_path: Path, run_id: str) -> list[LlmCall]:
    path = Workspace(tmp_path).attach_trace_file
    for _ in range(20):  # the writer flushes on completion; give it a moment
        if path.is_file() and any(run_id in line for line in path.read_text("utf-8").splitlines()):
            break
        time.sleep(0.05)
    if not path.is_file():
        return []
    lines = [line for line in path.read_text("utf-8").splitlines() if line.strip()]
    events = [parse_trace_line(line) for line in lines]
    return [e for e in events if isinstance(e, LlmCall) and e.run_id == run_id]


def test_fastapi_asgi_interceptor(attach_server: Path) -> None:
    app = FastAPI()

    @app.post("/chat")
    async def chat(request: Request) -> dict[str, str]:
        data = await request.json()
        client = AsyncOpenAI(
            api_key="t",
            base_url="http://fake.local/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(_fake_llm)),
            max_retries=0,
        )
        resp = await client.chat.completions.create(
            model="fake",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": data["message"]},
            ],
        )
        return {"reply": resp.choices[0].message.content or ""}

    client = TestClient(app)

    # a per-request overlay header rewrites the outgoing prompt
    trialed = client.post(
        "/chat",
        json={"message": "warm up"},
        headers={"x-agentopt-overlay": FRENCH_OVERLAY, "x-agentopt-run-id": "fastapi-run"},
    )
    assert trialed.json()["reply"] == "bonjour"
    # captured under this request's run id; the trace shows the trialed prompt
    stamped = _stamped_calls(attach_server, "fastapi-run")
    assert stamped and stamped[0].messages[0].content == SYSTEM + " Reply in French."

    # a header-less request is untouched (default reply)
    plain = client.post("/chat", json={"message": "hi"})
    assert plain.json()["reply"] == "hi there"


def test_flask_wsgi_interceptor(attach_server: Path) -> None:
    app = Flask(__name__)

    @app.post("/chat")
    def chat():  # type: ignore[no-untyped-def]
        data = request.get_json()
        client = OpenAI(
            api_key="t",
            base_url="http://fake.local/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(_fake_llm)),
            max_retries=0,
        )
        resp = client.chat.completions.create(
            model="fake",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": data["message"]},
            ],
        )
        return jsonify({"reply": resp.choices[0].message.content})

    client = app.test_client()

    trialed = client.post(
        "/chat",
        json={"message": "warm up"},
        headers={"X-Agentopt-Overlay": FRENCH_OVERLAY, "X-Agentopt-Run-Id": "flask-run"},
    )
    assert trialed.get_json()["reply"] == "bonjour"
    stamped = _stamped_calls(attach_server, "flask-run")
    assert stamped and stamped[0].messages[0].content == SYSTEM + " Reply in French."

    plain = client.post("/chat", json={"message": "hi"})
    assert plain.get_json()["reply"] == "hi there"
