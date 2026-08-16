"""Attach mode: the user runs their own server with the wiretap registered
(register_server); agentopt does NOT boot it. It sends the trial overlay and a
per-request run id as headers, and reads this request's calls out of the shared
attach trace — no restart.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from agentopt.models.config import load_config
from agentopt.models.trace import LlmCall, parse_trace_line
from agentopt.runner import RunnerError, runner_for
from agentopt.wiretap.overlay import Overlay, OverlayRule

# A user server that registers the wiretap itself, then serves /chat. It reads
# the body then calls OpenAI, so the per-request context is active at the call.
AGENT_SERVER = '''
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from agentopt import register_server
from openai import OpenAI

register_server(project_root=os.getcwd())

SYSTEM = "You are a terse fitness coach."
PORT = int(os.environ["PORT"])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(length).decode() or "{}")
        if self.path.startswith("/v1/chat/completions"):
            system = next(m["content"] for m in body["messages"] if m["role"] == "system")
            reply = "bonjour" if "Reply in French" in system else "hi there"
            self._json({
                "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            })
        elif self.path == "/chat":
            client = OpenAI(api_key="test", base_url=f"http://127.0.0.1:{PORT}/v1")
            resp = client.chat.completions.create(model="fake", messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": body["message"]},
            ])
            self._json({"reply": resp.choices[0].message.content})

    def _json(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


def _wait_for(url: str) -> None:
    for _ in range(100):
        try:
            httpx.get(url, timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise AssertionError(f"server never came up at {url}")


def test_attach_trials_via_per_request_headers(tmp_path: Path) -> None:
    port = 8400 + (os.getpid() % 500)
    (tmp_path / "server.py").write_text(AGENT_SERVER, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(
        # no `start:` -> attach mode
        "run:\n"
        "  http:\n"
        f"    ready: http://127.0.0.1:{port}/health\n"
        f"    url: http://127.0.0.1:{port}/chat\n"
        '    body: \'{"message":"{input}"}\'\n'
        "    output: $.reply\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PORT"] = str(port)
    # the USER starts the server (agentopt does not)
    server = subprocess.Popen(  # noqa: S603
        [sys.executable, "server.py"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for(f"http://127.0.0.1:{port}/health")
        runner = runner_for(load_config(tmp_path / "agentopt.yaml"), tmp_path)

        # 1. plain request — captured into an isolated per-run trace by run id
        plain = runner.run("what's a good warm-up?", mode="eval")
        assert plain.output == "hi there"
        assert plain.trace_path.is_file()
        events = [
            parse_trace_line(line)
            for line in plain.trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        call = next(e for e in events if isinstance(e, LlmCall))
        assert call.messages[0].content == "You are a terse fitness coach."
        assert call.run_id == plain.run_id  # stamped with this request's id

        # 2. a per-request overlay header rewrites the prompt — no server restart
        overlay = Overlay(rules=[OverlayRule(
            entry_id="e", kind="literal",
            match_exact="You are a terse fitness coach.",
            new_template="You are a terse fitness coach. Reply in French.",
        )])
        trialed = runner.run("what's a good warm-up?", mode="eval", overlay=overlay)
        assert trialed.output == "bonjour"
        assert trialed.run_id != plain.run_id
    finally:
        server.terminate()
        server.wait(timeout=10)


def test_attach_reports_a_missing_server_clearly(tmp_path: Path) -> None:
    (tmp_path / "agentopt.yaml").write_text(
        "run:\n"
        "  http:\n"
        "    ready: http://127.0.0.1:9/health\n"
        "    url: http://127.0.0.1:9/chat\n",
        encoding="utf-8",
    )
    runner = runner_for(load_config(tmp_path / "agentopt.yaml"), tmp_path)
    with pytest.raises(RunnerError, match="no server reachable"):
        runner.run("hi", mode="eval")
