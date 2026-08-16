"""The universal seams: ``expose`` and ``instrument``.

``instrument()`` captures a model call from a provider the auto-wiretap does
NOT patch (a plain function — no openai/anthropic SDK); the captured prompt is
located in source and trialed via overlay exactly like an auto-captured call.
``expose()`` normalizes output (sync or async agent).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentopt import expose, instrument, instrument_async
from agentopt.manifest import build_manifest
from agentopt.manifest.locator import PromptLocator
from agentopt.models.trace import LlmCall, parse_trace_line
from agentopt.runner import PythonRunner
from agentopt.wiretap import Wiretap
from agentopt.wiretap.overlay import Overlay, OverlayRule

SYSTEM = "You are a terse fitness coach."

# A real agent: it calls a NON-SDK "provider" (a plain function) through
# instrument(), and exposes its entry. The auto-wiretap never sees this call.
AGENT_SOURCE = '''
from agentopt import expose, instrument

SYSTEM = "You are a terse fitness coach."


def _provider(messages):
    system = next(m["content"] for m in messages if m["role"] == "system")
    return "bonjour" if "Reply in French" in system else "hi there"


def _agent(input_text):
    return instrument({
        "model": "fake-1",
        "provider": "custom",
        "params": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": input_text or ""},
        ],
    }, _provider)


run = expose(_agent)
'''


def _fake_provider(messages: list[dict]) -> str:
    system = next(m["content"] for m in messages if m["role"] == "system")
    return "bonjour" if "Reply in French" in system else "hi there"


def _spec(user: str) -> dict:
    return {
        "model": "fake-1",
        "provider": "custom",
        "params": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    }


def _events(trace_path: Path) -> list:
    return [
        parse_trace_line(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_instrument_captures_locates_and_trials(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(AGENT_SOURCE, encoding="utf-8")
    runner = PythonRunner(tmp_path, "agent:run")

    # 1. plain run — instrument captured a call the wiretap never sees; expose
    #    returned the answer.
    plain = runner.run("warm up", mode="observe")
    assert plain.output == "hi there"
    events = _events(plain.trace_path)
    call = next(e for e in events if isinstance(e, LlmCall))
    assert call.provider == "custom"
    assert call.model == "fake-1"
    assert call.params.temperature == 0.2
    assert call.messages[0].content == SYSTEM
    assert call.response is not None and call.response.content == "hi there"

    # 2. the captured prompt is located back in the agent's source, editable
    manifest = build_manifest(tmp_path, events)
    entry = next(e for e in manifest.entries if e.editable and e.kind == "prompt")
    assert entry.file == "agent.py"
    assert len(PromptLocator(tmp_path).find(SYSTEM, "agent.py")) == 1

    # 3. an overlay rewrites the outgoing prompt — the provider sees it
    overlay = Overlay(
        rules=[
            OverlayRule(
                entry_id="e",
                kind="literal",
                match_exact=SYSTEM,
                new_template=SYSTEM + " Reply in French.",
            )
        ]
    )
    trialed = runner.run("warm up", mode="eval", overlay=overlay)
    assert trialed.output == "bonjour"


def test_black_box_runner_captures_output_without_a_trace(tmp_path: Path) -> None:
    (tmp_path / "agent.py").write_text(AGENT_SOURCE, encoding="utf-8")
    runner = PythonRunner(tmp_path, "agent:run", black_box=True)  # no wiretap

    result = runner.run("warm up", mode="observe")

    assert result.output == "hi there"  # output still captured
    assert not result.trace_path.is_file()  # no trace — engine reads zero events


def test_instrument_async(tmp_path: Path) -> None:
    import asyncio

    async def provider(messages: list[dict]) -> str:
        return _fake_provider(messages)

    async def scenario() -> str:
        return await instrument_async(_spec("warm up"), provider)

    with Wiretap(tmp_path, mode="observe") as run:
        output = asyncio.run(scenario())
    assert output == "hi there"
    call = next(e for e in _events(run.trace_path) if isinstance(e, LlmCall))
    assert call.messages[0].content == SYSTEM


def test_instrument_passthrough_outside_a_run() -> None:
    # no capture context (e.g. production) — instrument just calls through
    result = instrument(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        lambda messages: f"saw {len(messages)} messages",
    )
    assert result == "saw 1 messages"


def test_expose_coerces_sync_and_async() -> None:
    assert expose(lambda _i: 42)("x") == "42"
    assert expose(lambda _i: None)("x") == ""

    async def agent(_input: str | None) -> str:
        return "async answer"

    assert expose(agent)("x") == "async answer"


def test_expose_dispatches_on_surface_when_entry_accepts_it() -> None:
    # A two-arg entry receives the surface; a one-arg entry is called plainly.
    dispatch = expose(lambda i, surface: f"{surface or 'default'}:{i}")
    assert dispatch("switch split", "router") == "router:switch split"
    assert dispatch("hello") == "default:hello"

    one_arg = expose(lambda i: f"only:{i}")
    assert one_arg("hi", "router") == "only:hi"  # surface ignored, no crash


def test_expose_map_form_dispatches_and_declares_surfaces() -> None:
    from agentopt.instrumentation import declared_surface_names

    run = expose({"default": lambda i: f"coach:{i}", "router": lambda i: f"route:{i}"})
    assert run("hi", "router") == "route:hi"
    assert run("hi") == "coach:hi"  # no surface -> default
    assert run("hi", "unknown") == "coach:hi"  # unknown -> default
    assert declared_surface_names() == frozenset({"default", "router"})


def test_expose_map_form_without_default_raises() -> None:
    run = expose({"router": lambda i: f"route:{i}"})
    with pytest.raises(ValueError, match="no 'default' handler"):
        run("hi")
