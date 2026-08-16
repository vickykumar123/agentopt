"""Watch the wiretap capture an agent's LLM calls — no API key needed.

A tiny "agent" makes two OpenAI-SDK calls answered by a mock transport; the
wiretap records both to a trace file, which is printed at the end.

Run:  python examples/wiretap_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import httpx
from openai import OpenAI

from agentopt.wiretap import Wiretap


def fake_provider(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    last_message = body["messages"][-1]["content"]
    reply = "intent: refund" if "back" in last_message else f"echo: {last_message}"
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        },
    )


def run_agent(client: OpenAI, user_message: str) -> str:
    """The 'user agent' being observed: classify, then respond."""
    classification = client.chat.completions.create(
        model="demo-model",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You classify fan messages into intents."},
            {"role": "user", "content": user_message},
        ],
    )
    intent = classification.choices[0].message.content or "unknown"

    reply = client.chat.completions.create(
        model="demo-model",
        temperature=0.9,
        messages=[
            {"role": "system", "content": f"You reply warmly. Detected {intent}."},
            {"role": "user", "content": user_message},
        ],
    )
    return reply.choices[0].message.content or ""


def main() -> None:
    client = OpenAI(
        api_key="demo-key",
        http_client=httpx.Client(transport=httpx.MockTransport(fake_provider)),
    )

    with tempfile.TemporaryDirectory() as tmp:
        project_root = Path(tmp)
        with Wiretap(project_root, input_text="money back pls??") as run:
            answer = run_agent(client, "money back pls??")

        print(f"agent answered: {answer!r}\n")
        print(f"trace file: {run.trace_path.relative_to(project_root)}\n")
        for line in run.trace_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["kind"] == "run_meta":
                print(f"[{event['seq']}] run_meta   mode={event['mode']} input={event['input']!r}")
            else:
                site = event.get("call_site") or {}
                print(
                    f"[{event['seq']}] {event['kind']}   model={event['model']} "
                    f"temp={event['params'].get('temperature')} "
                    f"from {site.get('file')}:{site.get('line')} "
                    f"-> {event['response']['content']!r}"
                )


if __name__ == "__main__":
    main()
