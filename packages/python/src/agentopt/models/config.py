"""Typed configuration — the Python view of ``schemas/config.schema.json``.

``agentopt.yaml`` is user-owned: scaffolded by ``agentopt init`` and editable
by hand at any time. Loading uses ``yaml.safe_load`` exclusively.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_FILENAME = "agentopt.yaml"


class HttpTarget(BaseModel):
    """HTTP harness: agentopt boots the server (``start``) with the wiretap
    hook, waits for ``ready``, then sends each eval input as a request.
    Secrets go in ``${ENV_VAR}`` (read from the environment, never the yaml);
    the eval input is substituted for ``{input}``."""

    model_config = ConfigDict(extra="forbid")

    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    output: str | None = None
    start: str | None = None
    ready: str | None = None
    ready_timeout_ms: int = Field(default=30_000, ge=1000)
    # Per-request bound on the agent HTTP call — a hung endpoint must not stall the
    # whole run. Generous by default (multi-step/tool agents can be slow).
    request_timeout_ms: int = Field(default=120_000, ge=1000)


class RunTarget(BaseModel):
    """How agentopt invokes the user's agent. Exactly one style must be set."""

    model_config = ConfigDict(extra="forbid")

    command: str | None = None
    http: HttpTarget | None = None
    python: str | None = None

    @model_validator(mode="after")
    def _exactly_one_style(self) -> RunTarget:
        styles = [v for v in (self.command, self.http, self.python) if v is not None]
        if len(styles) != 1:
            raise ValueError("run must set exactly one of: command, http, python")
        return self


class ImproverConfig(BaseModel):
    """Everything about the improver: which models power it and how it loops."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["claude-code", "codex", "cloud", "ollama", "auto"] = "auto"
    optimizer: str | None = None
    delegate: Literal["claude-code", "codex"] | None = Field(
        default=None,
        description="Run the tier-4 CODE DELEGATE (the agent that edits files) on a "
        "specific coding CLI, independent of `backend` — e.g. optimizer on the Claude "
        "subscription but code edits with delegate: codex (ChatGPT/Codex subscription). "
        "Unset = derive the delegate from `backend`.",
    )
    judge: str | None = None
    judge_command: str | None = None
    """A command that scores outputs instead of the built-in LLM judge — the
    project's own gate/scorers. Reads a JSON payload on stdin, prints a JSON
    {score} on stdout. Makes scoring architecture-insensitive."""
    judge_samples: int = Field(
        default=3,
        ge=1,
        description="Self-consistency for the built-in LLM judge: when a grading "
        "lands near the pass threshold (ambiguous), re-sample the judge up to this "
        "many times and take the median, so one noisy verdict can't flip a decision. "
        "Confident gradings still cost one call. Set 1 to disable.",
    )
    judge_backend: Literal["claude-code", "codex", "cloud", "ollama"] | None = Field(
        default=None,
        description="Run the JUDGE on a SEPARATE backend from the optimizer/delegate "
        "— e.g. keep the Claude subscription (backend: claude-code) for proposing and "
        "coding fixes, but grade with a cloud model (judge_backend: cloud, judge: a "
        "Fireworks/OpenAI-compatible model). Same credential rules as `backend`. Unset "
        "= the judge shares the main backend. Ignored when judge_command is set.",
    )
    judge_base_url: str | None = Field(
        default=None,
        description="For a judge_backend: cloud judge, the OpenAI-compatible endpoint "
        "to grade against (Fireworks, Together, Groq, OpenRouter, local vLLM, ...). "
        "Provider-generic. Falls back to OPENAI_BASE_URL, then OpenAI's default.",
    )
    judge_api_key_env: str | None = Field(
        default=None,
        description="The NAME of the env var holding the cloud judge's API key (the "
        "secret stays in the environment; this references it) — e.g. FIREWORKS_API_KEY. "
        "Falls back to OPENAI_API_KEY.",
    )
    simulator: str | None = None
    iterations: int = Field(default=5, ge=1, description="Improvement attempts per improve run.")
    runs_per_check: Literal["auto"] | int = Field(
        default="auto", description="Agent runs per eval input (fights randomness)."
    )
    eval_concurrency: int = Field(
        default=4,
        ge=1,
        description="How many agent runs to score concurrently. Scoring is the "
        "dominant cost and its runs are independent, so a bounded pool cuts "
        "wall-clock ~Nx. Peak memory scales with this cap (~N runs in flight), so "
        "keep it modest on constrained/unstable hardware. 1 = fully serial.",
    )
    min_effect_pp: float = Field(
        default=1.0,
        ge=0,
        description="The minimum paired net gain (percentage points) a change must "
        "clear to be accepted — the 'noise floor' (#7). Lower it to let smaller real "
        "gains land (at the risk of accepting noise); raise it to demand bigger wins.",
    )
    reliability_regression_tolerance: float = Field(
        default=0.02,
        ge=0,
        le=1,
        description="How far candidate execution reliability may fall below the "
        "baseline (as a fraction) before the change is rejected on reliability — the "
        "reliability gate runs BEFORE quality. Raise it for an agent whose runs are "
        "legitimately flaky; lower it to demand rock-steady execution.",
    )
    judge_concurrency: int = Field(
        default=1,
        ge=1,
        description="How many judge gradings to run concurrently. The judge makes "
        "independent, stateless calls, so this is safe; it speeds up an async/HTTP "
        "judge (a cloud model), not the blocking claude-code CLI judge. 1 = serial.",
    )
    stop_when_stuck: int = Field(
        default=2, ge=1, description="Give up after this many fruitless iterations."
    )
    explore: int = Field(
        default=3,
        ge=1,
        description="Diverse candidates generated and probed per cluster each round.",
    )
    require_holdout_gain: bool = Field(
        default=False,
        description="Require unseen-data (holdout) gain, not just non-regression.",
    )
    max_versions_per_run: int = Field(
        default=5,
        ge=1,
        description="Stop after creating this many versions (applied fixes) in one run.",
    )
    generate_challenges: bool = Field(
        default=False,
        description="When every check passes, invent adversarial challenges and keep "
        "improving instead of stopping (evolve mode).",
    )
    delegate_timeout_seconds: int = Field(
        default=1800,
        ge=30,
        description="Wall-clock budget for ONE code-delegate attempt. The delegate "
        "explores and edits a real repo, so a complex multi-file fix can exceed the old "
        "10-min cap; raise it for large codebases. On expiry the attempt is killed and "
        "its partial edits reverted (never applied unverified).",
    )
    code_attempts: int = Field(
        default=1,
        ge=1,
        description="#8: stochastic multi-attempt per failing cluster in code-only "
        "mode. The delegate is non-deterministic, so a rejected attempt may become a "
        "different, working fix on retry. >1 trades cost for robustness; default 1 keeps "
        "the current single-attempt behavior.",
    )


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    production: str | None = None
    allowed: list[str] = Field(default_factory=list)
    never: list[str] = Field(default_factory=list)


class CodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class WebsearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    provider: str = "auto"


class BudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tokens_per_run: int = Field(
        default=500_000,
        ge=1000,
        description="Hard cap on agent tokens consumed by one improve run.",
    )


class PrivacyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    redact: list[str] = Field(default_factory=list)
    retention_days: int = Field(default=30, ge=1)


class AgentoptConfig(BaseModel):
    """The complete ``agentopt.yaml``."""

    model_config = ConfigDict(extra="forbid")

    run: RunTarget
    improver: ImproverConfig = Field(default_factory=ImproverConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    code: CodeConfig = Field(default_factory=CodeConfig)
    protected: list[str] = Field(default_factory=list)
    websearch: WebsearchConfig = Field(default_factory=WebsearchConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


def load_config(path: Path) -> AgentoptConfig:
    """Load and validate an ``agentopt.yaml`` file."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} does not contain a YAML mapping")
    return AgentoptConfig.model_validate(raw)
