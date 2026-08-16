"""The ``run.http`` harness: output extraction, config, and a full integration
run against a real subprocess HTTP server booted with the wiretap hook —
request -> response -> trace -> overlay.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentopt.models.config import HttpTarget, load_config
from agentopt.models.trace import LlmCall, parse_trace_line
from agentopt.runner import HttpRunner, RunnerError, extract_path, runner_for
from agentopt.wiretap.overlay import Overlay, OverlayRule


class TestExtractPath:
    body = json.dumps({"reply": "hello there", "choices": [{"message": {"content": "nested"}}]})

    def test_dot_and_bracket_paths(self) -> None:
        assert extract_path(self.body, "$.reply") == "hello there"
        assert extract_path(self.body, "reply") == "hello there"
        assert extract_path(self.body, "$.choices[0].message.content") == "nested"

    def test_none_for_non_json_or_missing_path(self) -> None:
        # A response-schema mismatch is a structured invalid_response, not an
        # empty answer — so extract_path signals absence with None.
        assert extract_path("plain text", "$.reply") is None
        assert extract_path(self.body, "$.missing.deep") is None

    def test_serializes_non_string_leaves(self) -> None:
        assert extract_path(json.dumps({"n": 42}), "n") == "42"


class TestConfig:
    def test_http_object_parses_with_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "agentopt.yaml").write_text(
            "run:\n"
            "  http:\n"
            "    start: python server.py\n"
            "    url: http://127.0.0.1:8123/chat\n"
            "    headers:\n"
            '      authorization: "Bearer ${TOK}"\n'
            '    body: \'{"message":"{input}"}\'\n'
            "    output: $.reply\n",
            encoding="utf-8",
        )
        config = load_config(tmp_path / "agentopt.yaml")
        assert config.run.http is not None
        assert config.run.http.method == "POST"  # default
        assert config.run.http.headers["authorization"] == "Bearer ${TOK}"
        assert isinstance(runner_for(config, tmp_path), HttpRunner)

    def test_needs_a_start_command_for_now(self, tmp_path: Path) -> None:
        runner = HttpRunner(tmp_path, HttpTarget(url="http://127.0.0.1:9/x"))
        with pytest.raises(RunnerError, match="start"):
            runner.run("hi", mode="eval")


# A self-contained agent server: a threaded HTTP server hosting both the /chat
# endpoint and a fake OpenAI endpoint. /chat calls OpenAI (captured by the
# wiretap) with a literal system prompt the overlay can rewrite.
AGENT_SERVER = '''
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from openai import OpenAI

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
            payload = {
                "id": "x", "object": "chat.completion", "created": 0, "model": "fake",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            }
            self._json(payload)
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


class TestIntegration:
    def test_boot_request_trace_and_overlay(self, tmp_path: Path) -> None:
        port = 8200 + (os.getpid() % 500)
        (tmp_path / "server.py").write_text(AGENT_SERVER, encoding="utf-8")
        (tmp_path / "agentopt.yaml").write_text(
            "run:\n"
            "  http:\n"
            "    start: python server.py\n"
            f"    ready: http://127.0.0.1:{port}/health\n"
            f"    url: http://127.0.0.1:{port}/chat\n"
            '    body: \'{"message":"{input}"}\'\n'
            "    output: $.reply\n",
            encoding="utf-8",
        )
        os.environ["PORT"] = str(port)
        try:
            config = load_config(tmp_path / "agentopt.yaml")
            runner = runner_for(config, tmp_path)
            try:
                # 1. plain run — output from the JSON response
                plain = runner.run("what's a good warm-up?", mode="eval")
                assert plain.exit_code == 0
                assert plain.output == "hi there"

                # 2. the LLM call captured into an isolated per-run trace
                assert plain.trace_path.is_file()
                events = [parse_trace_line(line) for line in
                          plain.trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                call = next(e for e in events if isinstance(e, LlmCall))
                assert call.messages[0].content == "You are a terse fitness coach."
                assert call.response is not None and call.response.content == "hi there"

                # 3. an overlay restarts the server and rewrites the outgoing prompt
                overlay = Overlay(rules=[OverlayRule(
                    entry_id="e", kind="literal",
                    match_exact="You are a terse fitness coach.",
                    new_template="You are a terse fitness coach. Reply in French.",
                )])
                trialed = runner.run("what's a good warm-up?", mode="eval", overlay=overlay)
                assert trialed.output == "bonjour"
            finally:
                runner.close()
        finally:
            os.environ.pop("PORT", None)

    def test_non_2xx_response_is_an_agent_failure(self, tmp_path: Path) -> None:
        port = 8700 + (os.getpid() % 200)
        # A server that always 500s with a plausible fallback body — the exact
        # trap: without a status check this reads as the agent's answer.
        (tmp_path / "server.py").write_text(
            "import json, os\n"
            "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
            "class H(BaseHTTPRequestHandler):\n"
            "    def log_message(self, *a): pass\n"
            "    def do_GET(self):\n"
            "        self.send_response(200); self.end_headers(); self.wfile.write(b'ok')\n"
            "    def do_POST(self):\n"
            "        self.send_response(500)\n"
            "        self.send_header('content-type', 'application/json'); self.end_headers()\n"
            "        self.wfile.write(json.dumps({'reply': 'fallback'}).encode())\n"
            f"HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n",
            encoding="utf-8",
        )
        (tmp_path / "agentopt.yaml").write_text(
            "run:\n"
            "  http:\n"
            "    start: python server.py\n"
            f"    ready: http://127.0.0.1:{port}/health\n"
            f"    url: http://127.0.0.1:{port}/chat\n"
            '    body: \'{"message":"{input}"}\'\n'
            "    output: $.reply\n",
            encoding="utf-8",
        )
        config = load_config(tmp_path / "agentopt.yaml")
        runner = runner_for(config, tmp_path)
        try:
            result = runner.run("hello", mode="eval")
            assert result.exit_code == 1
            assert "HTTP 500" in result.output
            assert result.output != "fallback"  # the 500 body was NOT accepted
        finally:
            runner.close()


def test_http_failure_category_classification() -> None:
    import httpx

    from agentopt.runner import RunnerError, _http_failure_category

    assert _http_failure_category(httpx.TimeoutException("t")) == "timeout"
    server_500 = RunnerError("agent server returned HTTP 500: x")
    assert _http_failure_category(server_500) == "provider_error"
    assert _http_failure_category(httpx.ConnectError("refused")) == "provider_error"
    assert _http_failure_category(ValueError("weird")) == "runner_error"
