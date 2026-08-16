"""agentopt — self-improving loop for your existing AI agent.

Write what your agent *should* do in GOALS.md; agentopt measures the gap and
closes it: prompts, LLM parameters, tool descriptions, workflow, and a learned
playbook. No model training required.

https://github.com/vickykumar123/agentopt
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.0.1"

__all__ = ["InstrumentSpec", "expose", "instrument", "instrument_async", "register_server"]

# Lazy public API (PEP 562). Importing `agentopt` — or any leaf submodule such as
# `agentopt.hashing` — must NOT eagerly pull in the instrumentation/wiretap stack
# (which drags pydantic, asyncio, ...). Each symbol is imported only on first
# reference, so a consumer that just wants a stdlib-only utility never pays for
# the heavy stack, and a broken optional dependency can't break a leaf import.
# (The home module is `instrumentation`, not `instrument`, so the `instrument`
# function never clashes with a same-named submodule attribute — which would
# otherwise shadow it here.)
_LAZY_EXPORTS = {
    "InstrumentSpec": "agentopt.instrumentation",
    "expose": "agentopt.instrumentation",
    "instrument": "agentopt.instrumentation",
    "instrument_async": "agentopt.instrumentation",
    "register_server": "agentopt.wiretap.session",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache: subsequent access is a plain attribute lookup
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])


if TYPE_CHECKING:  # static analysers/IDEs still resolve the real symbols
    from agentopt.instrumentation import (
        InstrumentSpec as InstrumentSpec,
    )
    from agentopt.instrumentation import (
        expose as expose,
    )
    from agentopt.instrumentation import (
        instrument as instrument,
    )
    from agentopt.instrumentation import (
        instrument_async as instrument_async,
    )
    from agentopt.wiretap.session import register_server as register_server
