"""The eval-run cache (#5) — cheaper re-measures + resume-after-kill."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agentopt.evals import compile_goals
from agentopt.evals.engine import EvalEngine
from agentopt.evals.eval_cache import CachedRun, EvalCache, code_fingerprint
from agentopt.goals import parse_goals
from agentopt.runner import AgentRunner, RunResult
from agentopt.wiretap.context import Mode
from agentopt.wiretap.overlay import Overlay

GOALS = '## Examples\n- Input: "hi" -> mentions ok'


class _CountingRunner(AgentRunner):
    def __init__(self) -> None:
        super().__init__(Path("."))
        self.calls = 0

    def run(
        self,
        input_text: str | None,
        mode: Mode,
        overlay: Overlay | None = None,
        surface: str | None = None,
    ) -> RunResult:
        self.calls += 1
        return RunResult(
            run_id="r", trace_path=Path("nope"), output="ok", exit_code=0, tools=[], route=None
        )


def test_persists_and_reloads_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "runs.jsonl"
    key = EvalCache.key_of("fp", "default", "hi", 0)
    EvalCache(path).put(key, CachedRun(output="x", agent_failed=False, tools=[], route=None))
    # A brand-new instance (as a restarted process would build) reads from disk.
    assert EvalCache(path).get(key) is not None
    assert EvalCache(path).get(key).output == "x"  # type: ignore[union-attr]


def test_replays_cached_runs_on_remeasure(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache" / "runs.jsonl")
    runner = _CountingRunner()
    suite = compile_goals(parse_goals(GOALS))

    EvalEngine(runner, repeats=2, cache=cache, code_fingerprint="fp1").run_suite(suite)
    assert runner.calls == 2  # both repeats actually ran

    EvalEngine(runner, repeats=2, cache=cache, code_fingerprint="fp1").run_suite(suite)
    assert runner.calls == 2  # identical fingerprint → fully cached, no new runs


def test_changed_fingerprint_misses_cache(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache" / "runs.jsonl")
    runner = _CountingRunner()
    suite = compile_goals(parse_goals(GOALS))

    EvalEngine(runner, repeats=1, cache=cache, code_fingerprint="fpA").run_suite(suite)
    assert runner.calls == 1
    EvalEngine(runner, repeats=1, cache=cache, code_fingerprint="fpB").run_suite(suite)
    assert runner.calls == 2  # code changed → re-ran


def test_no_fingerprint_disables_caching(tmp_path: Path) -> None:
    cache = EvalCache(tmp_path / "cache" / "runs.jsonl")
    runner = _CountingRunner()
    suite = compile_goals(parse_goals(GOALS))

    EvalEngine(runner, repeats=1, cache=cache, code_fingerprint="").run_suite(suite)
    EvalEngine(runner, repeats=1, cache=cache, code_fingerprint="").run_suite(suite)
    assert runner.calls == 2  # no caching → ran both times


def _git_repo(root: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)  # noqa: S603,S607

    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    git("init", "-q")
    git("config", "user.email", "t@t.t")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-q", "-m", "init")


def test_fingerprint_tracks_untracked_source(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    base = code_fingerprint(tmp_path)
    assert base != ""  # real repo → a real fingerprint

    (tmp_path / "entry.py").write_text("V = 1\n", encoding="utf-8")  # untracked agent source
    with_file = code_fingerprint(tmp_path)
    assert with_file != base  # git diff HEAD misses it, but we fold it in

    (tmp_path / "entry.py").write_text("V = 2\n", encoding="utf-8")  # same file, new content
    assert code_fingerprint(tmp_path) != with_file  # stale-cache bug is fixed


def test_fingerprint_ignores_workspace_and_gitignored(tmp_path: Path) -> None:
    _git_repo(tmp_path)
    base = code_fingerprint(tmp_path)

    ws = tmp_path / ".agentopt" / "logs"
    ws.mkdir(parents=True)
    (ws / "run.log").write_text("noise\n", encoding="utf-8")
    (tmp_path / ".agentopt" / "cache.jsonl").write_text("{}\n", encoding="utf-8")
    assert code_fingerprint(tmp_path) == base  # workspace churn must not bust the cache

    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    subprocess.run(  # noqa: S603
        ["git", "add", ".gitignore"], cwd=tmp_path, check=True, capture_output=True  # noqa: S607
    )
    subprocess.run(  # noqa: S603
        ["git", "commit", "-q", "-m", "ignore"], cwd=tmp_path, check=True, capture_output=True  # noqa: S607
    )
    after_ignore = code_fingerprint(tmp_path)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "out.js").write_text("generated\n", encoding="utf-8")
    assert code_fingerprint(tmp_path) == after_ignore  # gitignored artifact excluded
