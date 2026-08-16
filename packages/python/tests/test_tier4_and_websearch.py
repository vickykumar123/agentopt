"""Tier-4 delegated code fixes (evidence-gated, git-required) and the
fail-soft web search policy.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from agentopt.evals import compile_goals
from agentopt.gateway.base import ChatMessage, GatewayError
from agentopt.gateway.websearch import search_guidance
from agentopt.goals import parse_goals
from agentopt.improver.tier4 import (
    ClaudeCliAgent,
    CodeDelegate,
    CodexCliAgent,
    DeepAgentsAgent,
    build_code_agent,
)
from agentopt.models.config import ImproverConfig, load_config
from agentopt.runner import AgentRunner, runner_for

BROKEN_AGENT = """
import httpx
from openai import OpenAI

PROMPT = "You are a billing assistant."


def run(input_text):
    answer = "We have plans."
    payload = {
        "id": "c", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    client = OpenAI(
        api_key="k",
        http_client=httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ),
        max_retries=0,
    )
    out = client.chat.completions.create(
        model="m",
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": input_text or ""}],
    )
    return out.choices[0].message.content
"""

GOALS = '## Examples\n- Input: "how much is premium" → mentions $9.99\n'

FIXING_DELEGATE = f"""#!{sys.executable}
import sys
from pathlib import Path

path = Path("billing_agent.py")
source = path.read_text(encoding="utf-8")
path.write_text(source.replace('"We have plans."', '"Premium is $9.99 per month."'),
                encoding="utf-8")
print('{{"is_error": false, "result": "fixed"}}')
"""

USELESS_DELEGATE = f"""#!{sys.executable}
print('{{"is_error": false, "result": "nothing to do"}}')
"""

WRONG_DELEGATE = f"""#!{sys.executable}
from pathlib import Path

path = Path("billing_agent.py")
source = path.read_text(encoding="utf-8")
path.write_text(source.replace('"We have plans."', '"Plans start soon."'), encoding="utf-8")
print('{{"is_error": false, "result": "edited but wrong"}}')
"""

WRONG_UNTRACKED_DELEGATE = f"""#!{sys.executable}
from pathlib import Path

path = Path("billing_agent.py")
path.write_text(path.read_text(encoding="utf-8").replace('"We have plans."', '"Plans soon."'),
                encoding="utf-8")
Path("notes.txt").write_text("DELEGATE OVERWROTE THIS\\n", encoding="utf-8")
print('{{"is_error": false, "result": "edited untracked too"}}')
"""


def _git(project: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 -- no shell
        ["git", *args],  # noqa: S607 -- fixed git command
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def git_project(tmp_path: Path) -> Path:
    (tmp_path / "billing_agent.py").write_text(BROKEN_AGENT, encoding="utf-8")
    (tmp_path / "agentopt.yaml").write_text(
        'run:\n  python: "billing_agent:run"\n', encoding="utf-8"
    )
    (tmp_path / "GOALS.md").write_text(GOALS, encoding="utf-8")
    (tmp_path / ".gitignore").write_text(
        "fake_claude.py\n.agentopt/\n__pycache__/\n", encoding="utf-8"
    )
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")
    return tmp_path


def _fake_delegate(tmp_path: Path, script: str) -> str:
    target = tmp_path / "fake_claude.py"
    target.write_text(script, encoding="utf-8")
    return f"{sys.executable} {target}"


def _delegate_for(project: Path, executable: str) -> CodeDelegate:
    config = load_config(project / "agentopt.yaml")
    runner = runner_for(config, project)
    return CodeDelegate(project, runner, judge=None, repeats=1, executable=executable)


def _failing_cases(project: Path) -> list:
    from agentopt.evals.engine import EvalEngine
    from agentopt.models.config import load_config as load
    from agentopt.runner import runner_for as rf

    config = load(project / "agentopt.yaml")
    suite = compile_goals(parse_goals(GOALS))
    scorecard = EvalEngine(rf(config, project), repeats=1).run_suite(suite)
    return [r for r in scorecard.cases if r.status == "failed"]


class _CountingRunner(AgentRunner):
    """Counts agent runs, so a test can prove the gate skipped a baseline re-measure."""

    def __init__(self, inner: AgentRunner) -> None:
        super().__init__(inner._project_root)
        self._inner = inner
        self.runs = 0
        self.parallel_safe = inner.parallel_safe

    def run(self, input_text, mode, overlay=None, surface=None):  # type: ignore[override,no-untyped-def]
        self.runs += 1
        return self._inner.run(input_text, mode, overlay, surface)


class _SubprocessDelegate(CodeDelegate):
    """Test double: runs the fake delegate script instead of the claude CLI."""

    def __init__(self, project: Path, runner: object, command: str) -> None:
        super().__init__(project, runner, judge=None, repeats=1, executable="claude")  # type: ignore[arg-type]
        self._command = command

    def _run_delegate(self, failures: list) -> str | None:  # type: ignore[override]
        completed = subprocess.run(  # noqa: S603 -- test-controlled command
            self._command.split(),
            cwd=self._project_root,
            capture_output=True,
            text=True,
        )
        return None if completed.returncode == 0 else "delegate failed"


class TestTier4:
    def test_successful_code_fix_is_verified_and_versioned(
        self, git_project: Path, tmp_path: Path
    ) -> None:
        command = _fake_delegate(git_project, FIXING_DELEGATE)
        config = load_config(git_project / "agentopt.yaml")
        runner = runner_for(config, git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        suite = compile_goals(parse_goals(GOALS))
        result = delegate.attempt(suite, _failing_cases(git_project))

        assert result.attempted and result.accepted
        assert result.version == "v1"
        assert "billing_agent.py" in (result.changed_files or [])
        assert "$9.99" in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_params_change_is_applied_to_source_and_versioned(
        self, git_project: Path, tmp_path: Path
    ) -> None:
        command = _fake_delegate(git_project, FIXING_DELEGATE)
        config = load_config(git_project / "agentopt.yaml")
        runner = runner_for(config, git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        from agentopt.improver.tier4 import ParamTarget

        suite = compile_goals(parse_goals(GOALS))
        result = delegate.apply_params(
            suite,
            {"temperature": 0.3},
            ParamTarget(file="billing_agent.py", line=3, prompt_snippet="You are a billing agent."),
        )

        assert result.attempted and result.accepted
        assert result.version == "v1"
        assert "params" in result.reason
        assert "$9.99" in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_useless_change_is_reverted(self, git_project: Path) -> None:
        command = _fake_delegate(git_project, USELESS_DELEGATE)
        config = load_config(git_project / "agentopt.yaml")
        runner = runner_for(config, git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        suite = compile_goals(parse_goals(GOALS))
        result = delegate.attempt(suite, _failing_cases(git_project))

        assert result.attempted and not result.accepted
        assert "changed no files" in result.reason

    def test_runs_on_dirty_tree_and_reverts_to_pre_delegate_state(
        self, git_project: Path
    ) -> None:
        # A prior uncommitted change to the SAME file the delegate will edit. The
        # old rule refused tier-4 outright; a HEAD-based revert would wipe this.
        src = (git_project / "billing_agent.py").read_text(encoding="utf-8")
        prior = src + "\n# PRIOR AGENTOPT CHANGE\n"
        (git_project / "billing_agent.py").write_text(prior, encoding="utf-8")
        command = _fake_delegate(git_project, WRONG_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        result = delegate.attempt(compile_goals(parse_goals(GOALS)), _failing_cases(git_project))

        assert result.attempted  # no longer blocked by a dirty tree
        assert not result.accepted  # wrong edit → reverted
        restored = (git_project / "billing_agent.py").read_text(encoding="utf-8")
        assert "# PRIOR AGENTOPT CHANGE" in restored  # prior change survives
        assert "We have plans." in restored  # delegate edit gone
        assert "Plans start soon." not in restored

    def test_reverts_a_delegate_edit_to_an_untracked_file(self, git_project: Path) -> None:
        # An untracked file present before the delegate (e.g. a run harness / entry).
        (git_project / "notes.txt").write_text("original untracked content\n", encoding="utf-8")
        command = _fake_delegate(git_project, WRONG_UNTRACKED_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        result = delegate.attempt(compile_goals(parse_goals(GOALS)), _failing_cases(git_project))

        assert not result.accepted  # no gain → reverted
        # The untracked file is restored to its pre-delegate content (git stash
        # create misses it; the content snapshot does not).
        assert (git_project / "notes.txt").read_text(encoding="utf-8") == (
            "original untracked content\n"
        )
        assert "We have plans." in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_accepted_edit_snapshots_only_the_delegate_file(self, git_project: Path) -> None:
        # A prior uncommitted change to a DIFFERENT tracked file.
        helper = git_project / "helper.py"
        helper.write_text("VALUE = 'v0'\n", encoding="utf-8")
        _git(git_project, "-c", "user.name=t", "-c", "user.email=t@t", "add", "helper.py")
        _git(git_project, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "helper")
        helper.write_text("VALUE = 'prior'\n", encoding="utf-8")  # uncommitted prior change
        command = _fake_delegate(git_project, FIXING_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        result = delegate.attempt(compile_goals(parse_goals(GOALS)), _failing_cases(git_project))

        assert result.accepted
        assert result.changed_files == ["billing_agent.py"]  # only the delegate's file
        assert helper.read_text(encoding="utf-8") == "VALUE = 'prior'\n"  # prior survives

    def test_reverts_partial_edits_when_the_delegate_errors(self, git_project: Path) -> None:
        # The delegate edits a file, then reports failure (as a timeout does) — its
        # partial edit must be reverted, not left as unverified residue.
        class _EditThenFail(_SubprocessDelegate):
            def _run_delegate(self, prompt: str) -> str | None:  # type: ignore[override]
                path = self._project_root / "billing_agent.py"
                path.write_text(
                    path.read_text(encoding="utf-8").replace('"We have plans."', '"Plans soon."'),
                    encoding="utf-8",
                )
                return "delegate timed out after 1800s"

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _EditThenFail(git_project, runner, "true")

        result = delegate.attempt(compile_goals(parse_goals(GOALS)), _failing_cases(git_project))

        assert not result.accepted
        assert "timed out" in result.reason and "reverted" in result.reason
        assert "We have plans." in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_accepts_good_fix_when_agentopt_dir_not_gitignored(self, tmp_path: Path) -> None:
        # A project that does NOT gitignore .agentopt/: the B2 journal must not be
        # seen as a delegate-created protected file (B6).
        (tmp_path / "billing_agent.py").write_text(BROKEN_AGENT, encoding="utf-8")
        (tmp_path / "agentopt.yaml").write_text(
            'run:\n  python: "billing_agent:run"\n', encoding="utf-8"
        )
        (tmp_path / "GOALS.md").write_text(GOALS, encoding="utf-8")
        # deliberately NO .agentopt/ entry in .gitignore
        (tmp_path / ".gitignore").write_text("fake_claude.py\n__pycache__/\n", encoding="utf-8")
        _git(tmp_path, "init", "-b", "main")
        _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
        _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "init")

        command = _fake_delegate(tmp_path, FIXING_DELEGATE)
        runner = runner_for(load_config(tmp_path / "agentopt.yaml"), tmp_path)
        delegate = _SubprocessDelegate(tmp_path, runner, command)

        # attempt_for_cluster writes the B2 journal before the delegate runs.
        result = delegate.attempt_for_cluster(
            compile_goals(parse_goals(GOALS)), None, _failing_cases(tmp_path), "billing_agent.py"
        )

        assert "protected files" not in result.reason  # journal didn't self-trip
        assert result.accepted
        assert result.changed_files == ["billing_agent.py"]  # only the real edit

    def test_rejects_edit_to_a_config_protected_file(self, git_project: Path) -> None:
        # B7: a delegate that edits a protected path (e.g. the harness) is rejected
        # and reverted, even if it also made a real product fix.
        (git_project / "harness.py").write_text("H = 1\n", encoding="utf-8")

        class _TamperHarness(_SubprocessDelegate):
            def _run_delegate(self, prompt: str) -> str | None:  # type: ignore[override]
                agent = self._project_root / "billing_agent.py"
                agent.write_text(
                    agent.read_text(encoding="utf-8").replace(
                        '"We have plans."', '"Premium is $9.99 per month."'
                    ),
                    encoding="utf-8",
                )
                (self._project_root / "harness.py").write_text(
                    "H = 2  # tampered\n", encoding="utf-8"
                )
                return None

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _TamperHarness(git_project, runner, "true")
        delegate._protected_paths = ("harness.py",)  # simulate config.protected

        result = delegate.attempt(compile_goals(parse_goals(GOALS)), _failing_cases(git_project))

        assert not result.accepted
        assert "protected files" in result.reason
        assert "tampered" not in (git_project / "harness.py").read_text(encoding="utf-8")
        assert "We have plans." in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_non_git_project_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "billing_agent.py").write_text(BROKEN_AGENT, encoding="utf-8")
        (tmp_path / "agentopt.yaml").write_text(
            'run:\n  python: "billing_agent:run"\n', encoding="utf-8"
        )
        delegate = _delegate_for(tmp_path, "claude")
        result = delegate.attempt(compile_goals(parse_goals(GOALS)), [])
        assert not result.attempted
        assert "git repository" in result.reason

    def test_available_reflects_whether_an_agent_is_wired(self, git_project: Path) -> None:
        wired = _SubprocessDelegate(
            git_project, runner_for(load_config(git_project / "agentopt.yaml"), git_project), "true"
        )
        assert wired.available() is True

    def test_cluster_code_candidate_accepts_a_structural_fix(self, git_project: Path) -> None:
        command = _fake_delegate(git_project, FIXING_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        suite = compile_goals(parse_goals(GOALS))
        result = delegate.attempt_for_cluster(
            suite, None, _failing_cases(git_project), "billing_agent.py"
        )

        assert result.attempted and result.accepted
        assert result.version == "v1"
        assert result.baseline_score == 0
        assert result.candidate_score == 100
        assert "$9.99" in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_reuses_a_passed_baseline_skipping_re_measure(self, git_project: Path) -> None:
        # The gate must reuse the caller's baseline instead of re-running the agent
        # for it — the double-baseline token saving.
        from agentopt.evals.engine import EvalEngine

        config = load_config(git_project / "agentopt.yaml")
        suite = compile_goals(parse_goals(GOALS))
        # One full baseline measurement (as the loop would), and its run count.
        ref = _CountingRunner(runner_for(config, git_project))
        pre = EvalEngine(ref, repeats=1).run_suite(suite)
        one_measure = ref.runs

        command = _fake_delegate(git_project, FIXING_DELEGATE)
        counter = _CountingRunner(runner_for(config, git_project))
        failures = [c for c in pre.cases if c.status == "failed"]
        result = _SubprocessDelegate(git_project, counter, command).attempt_for_cluster(
            suite, None, failures, "billing_agent.py", pre_baseline=(pre, None)
        )

        assert result.attempted and result.accepted  # still evaluates correctly
        # With the baseline reused, the gate runs the agent only for the CANDIDATE
        # (one_measure) + the probe (one run per failing case) — NOT a second full
        # baseline. Re-measuring the baseline would add another `one_measure`, which
        # this bound excludes.
        assert counter.runs <= one_measure + len(failures)

    def test_probe_rejects_a_non_fixing_edit_before_full_score(self, git_project: Path) -> None:
        # D: WRONG_DELEGATE edits the file but doesn't fix the failing case, so the
        # cheap probe rejects it (and reverts) without the full re-score.
        command = _fake_delegate(git_project, WRONG_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        suite = compile_goals(parse_goals(GOALS))
        result = delegate.attempt_for_cluster(
            suite, None, _failing_cases(git_project), "billing_agent.py"
        )

        assert result.attempted and not result.accepted
        assert "probe: no previously-failing case improved" in result.reason
        assert "We have plans." in (git_project / "billing_agent.py").read_text(encoding="utf-8")
        # A probe rejection is audited like a full-gate rejection: it leaves a
        # structured decision artifact (#2), so no reverted attempt is silent.
        assert result.decision_ref is not None
        reports = git_project / ".agentopt" / "reports"
        assert (reports / result.decision_ref).is_file()

    def test_probe_lets_a_fixing_edit_through_to_the_full_gate(self, git_project: Path) -> None:
        # D: a fix that satisfies the failing case improves it, so the probe passes
        # and the full significance gate accepts it.
        command = _fake_delegate(git_project, FIXING_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        suite = compile_goals(parse_goals(GOALS))
        result = delegate.attempt_for_cluster(
            suite, None, _failing_cases(git_project), "billing_agent.py"
        )

        assert result.attempted and result.accepted
        assert "probe:" not in result.reason  # passed the probe, went to the full gate
        assert "$9.99" in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_probe_executes_the_full_input_universe_for_a_null_input_case(
        self, git_project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # #1: a null-input rule grades over its surface's WHOLE input set, so the probe
        # must EXECUTE that full set (via run_plan) — not a single synthetic run — or a
        # null-input failure is compared against mismatched evidence.
        from datetime import datetime, timezone

        from agentopt.evals import engine as engine_mod
        from agentopt.evals.scorecard import CaseResult, Scorecard

        goals = (
            '# G\n\n## rules\n- Never say "oops"\n\n'
            '## examples\n- Input: "a" -> mentions x\n- Input: "b" -> mentions x\n'
        )
        train = compile_goals(parse_goals(goals))
        rule = next(c for c in train.cases if c.input is None)

        def _c(cid: str, status: str) -> CaseResult:
            return CaseResult(
                case_id=cid, source=cid, layer=1 if cid == rule.id else 2,
                runs=1, passed_runs=0 if status == "failed" else 1,
                score=0.0 if status == "failed" else 1.0,
                status=status, detail="",  # type: ignore[arg-type]
            )

        baseline_train = Scorecard(
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            score=0.0, measured_cases=len(train.cases), skipped_cases=0,
            cases=[_c(rule.id, "failed")]
            + [_c(c.id, "passed") for c in train.cases if c.input is not None],
        )

        captured: dict[str, object] = {}
        original = engine_mod.EvalEngine.run_suite

        def _spy(self, suite, run_plan=None):  # type: ignore[no-untyped-def]
            captured["run_plan"] = run_plan
            captured["graded"] = [c.id for c in suite.cases]
            return original(self, suite, run_plan)

        monkeypatch.setattr(engine_mod.EvalEngine, "run_suite", _spy)

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = CodeDelegate(git_project, runner, judge=None, repeats=1, executable="claude")
        baseline = delegate._capture_baseline()
        delegate._probe_failing_cases(train, baseline_train, [], baseline, "probe test")

        # Graded ONLY the failing null-input rule, but the run_plan covers the full
        # surface input set (all 3 cases) — like-for-like with the baseline evidence.
        assert captured["graded"] == [rule.id]
        run_plan = captured["run_plan"]
        assert run_plan is not None
        assert len(run_plan.cases) == len(train.cases)  # full input universe, not 1

    def test_cluster_code_candidate_reverts_a_non_helpful_edit(self, git_project: Path) -> None:
        command = _fake_delegate(git_project, WRONG_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        suite = compile_goals(parse_goals(GOALS))
        result = delegate.attempt_for_cluster(
            suite, None, _failing_cases(git_project), "billing_agent.py"
        )

        assert result.attempted and not result.accepted
        # WRONG_DELEGATE never satisfies the "$9.99" case, so it doesn't improve at
        # all — the cheap probe (D) rejects it before the full re-score. Either way
        # it is reverted.
        assert (
            "probe: no previously-failing case improved" in result.reason
            or "noise floor" in result.reason
            or "not significant" in result.reason
        )
        assert "We have plans." in (git_project / "billing_agent.py").read_text(encoding="utf-8")


class TestCrashSafeJournal:
    """B2: a crash mid-delegate must not leave unverified edits in the tree."""

    def test_clears_the_pending_marker_after_a_normal_attempt(self, git_project: Path) -> None:
        from agentopt.paths import Workspace

        command = _fake_delegate(git_project, FIXING_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        delegate.attempt(compile_goals(parse_goals(GOALS)), _failing_cases(git_project))

        # The finally ran: no interrupted-attempt marker is left behind.
        assert not Workspace(git_project).pending_attempt_file.exists()

    def test_recovers_an_interrupted_attempt(self, git_project: Path) -> None:
        import json

        from agentopt.paths import Workspace

        ws = Workspace(git_project)
        # Simulate a run killed mid-delegate: a pending marker plus the delegate's
        # unverified edit still in the tree (the finally never ran).
        agent = git_project / "billing_agent.py"
        agent.write_text(
            agent.read_text(encoding="utf-8").replace('"We have plans."', '"Plans soon."'),
            encoding="utf-8",
        )
        ws.root.mkdir(parents=True, exist_ok=True)
        ws.pending_attempt_file.write_text(
            json.dumps({"version": 1, "description": "x", "ref": "", "untracked": {}}),
            encoding="utf-8",
        )

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        reverted = _SubprocessDelegate(git_project, runner, "true").recover_interrupted()

        assert reverted == ["billing_agent.py"]
        assert "We have plans." in agent.read_text(encoding="utf-8")  # reverted
        assert not ws.pending_attempt_file.exists()  # marker cleared

    def test_restores_an_untracked_file_from_the_journal(self, git_project: Path) -> None:
        import json

        from agentopt.paths import Workspace

        ws = Workspace(git_project)
        (git_project / "entry.py").write_text("DELEGATE OVERWROTE THIS\n", encoding="utf-8")
        ws.root.mkdir(parents=True, exist_ok=True)
        ws.pending_attempt_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "description": "x",
                    "ref": "",
                    "untracked": {"entry.py": "original harness\n"},
                }
            ),
            encoding="utf-8",
        )

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        reverted = _SubprocessDelegate(git_project, runner, "true").recover_interrupted()

        assert "entry.py" in reverted
        assert (git_project / "entry.py").read_text(encoding="utf-8") == "original harness\n"
        assert not ws.pending_attempt_file.exists()

    def test_is_a_no_op_without_a_marker(self, git_project: Path) -> None:
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        assert _SubprocessDelegate(git_project, runner, "true").recover_interrupted() == []


class TestCodeOnlyLoop:
    """The architecture-insensitive loop: black-box run + score → code delegate."""

    def _loop(self, project: Path, delegate: CodeDelegate):  # type: ignore[no-untyped-def]
        from agentopt.improver.code_only import CodeOnlyLoop
        from agentopt.paths import Workspace

        runner = runner_for(load_config(project / "agentopt.yaml"), project)
        return CodeOnlyLoop(
            project_root=project,
            runner=runner,
            judge=None,
            repeats=1,
            tier4=delegate,
            ledger_path=Workspace(project).ledger_file,
            max_rounds=2,
            max_changes=5,
            plateau_stop=2,
        )

    def test_improves_a_black_box_agent_via_the_delegate(self, git_project: Path) -> None:
        command = _fake_delegate(git_project, FIXING_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        outcome = self._loop(git_project, delegate).run(compile_goals(parse_goals(GOALS)))

        assert len(outcome.accepted) == 1
        assert outcome.accepted[0].version == "v1"
        assert outcome.final_score == 100
        assert "$9.99" in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_retries_a_rejected_cluster_then_accepts(self, git_project: Path) -> None:
        # #8: first attempt is non-improving (rejected + reverted); the retry finds
        # the working fix. Mirrors the stochastic delegate.
        calls = {"n": 0}

        class _Flaky(_SubprocessDelegate):
            def _run_delegate(self, prompt: str) -> str | None:  # type: ignore[override]
                calls["n"] += 1
                path = self._project_root / "billing_agent.py"
                good = '"Premium is $9.99 per month."'
                replacement = good if calls["n"] >= 2 else '"Plans start soon."'
                path.write_text(
                    path.read_text(encoding="utf-8").replace('"We have plans."', replacement),
                    encoding="utf-8",
                )
                return None

        from agentopt.improver.code_only import CodeOnlyLoop
        from agentopt.paths import Workspace

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        outcome = CodeOnlyLoop(
            project_root=git_project,
            runner=runner,
            judge=None,
            repeats=1,
            tier4=_Flaky(git_project, runner, "true"),
            ledger_path=Workspace(git_project).ledger_file,
            max_rounds=1,
            max_changes=5,
            plateau_stop=2,
            max_attempts_per_cluster=2,
        ).run(compile_goals(parse_goals(GOALS)))

        assert calls["n"] == 2  # retried once after the first was rejected
        assert len(outcome.accepted) == 1  # the retry cleared the gate
        assert "$9.99" in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_applies_nothing_when_no_edit_improves_the_score(self, git_project: Path) -> None:
        command = _fake_delegate(git_project, WRONG_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        outcome = self._loop(git_project, delegate).run(compile_goals(parse_goals(GOALS)))

        assert len(outcome.accepted) == 0
        assert "We have plans." in (git_project / "billing_agent.py").read_text(encoding="utf-8")

    def test_aborts_when_most_agent_runs_fail(self, git_project: Path) -> None:
        # Passes the null-input preflight, but fails on real inputs (like
        # intermittent API errors) → B4 sees mostly-failed runs and aborts.
        (git_project / "billing_agent.py").write_text(
            "def run(input_text=None):\n"
            "    if input_text:\n        raise RuntimeError('API down')\n"
            "    return 'ok'\n",
            encoding="utf-8",
        )
        delegated = {"called": False}

        class _NoDelegate(_SubprocessDelegate):
            def _run_delegate(self, prompt: str) -> str | None:  # type: ignore[override]
                delegated["called"] = True
                return None

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        outcome = self._loop(git_project, _NoDelegate(git_project, runner, "true")).run(
            compile_goals(parse_goals(GOALS))
        )

        assert len(outcome.accepted) == 0
        assert any("agent runs failed" in n for n in outcome.notes)
        assert delegated["called"] is False  # aborted before ever calling the delegate

    def test_preflight_aborts_fast_with_the_real_error(self, git_project: Path) -> None:
        # A broken entry (raises on import/run) — preflight surfaces the error
        # immediately, before any baseline eval or delegation.
        (git_project / "billing_agent.py").write_text(
            "raise ImportError('No module named missing')\n"
            "def run(input_text=None):\n    return 'x'\n",
            encoding="utf-8",
        )
        delegated = {"called": False}

        class _NoDelegate(_SubprocessDelegate):
            def _run_delegate(self, prompt: str) -> str | None:  # type: ignore[override]
                delegated["called"] = True
                return None

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        outcome = self._loop(git_project, _NoDelegate(git_project, runner, "true")).run(
            compile_goals(parse_goals(GOALS))
        )

        assert len(outcome.accepted) == 0
        assert any("failed to run" in n for n in outcome.notes)
        assert any("missing" in n for n in outcome.notes)  # real error surfaced
        assert delegated["called"] is False


class TestPlaybookAndCodeDelegate:
    """The playbook ↔ code-delegate wiring: the delegate READS past lessons on
    every run, and an accepted code fix WRITES a new lesson (cross-run memory
    that code-only mode previously never produced)."""

    def test_feeds_past_playbook_lessons_into_the_delegate_prompt(
        self, git_project: Path
    ) -> None:
        # A pre-existing accepted lesson (BULLET format with an id marker).
        (git_project / "playbook.md").write_text(
            "# Playbook\n\n- Prefer $9.99 pricing <!-- id:abc -->\n", encoding="utf-8"
        )
        seen: dict[str, str] = {}

        class _Capturing(_SubprocessDelegate):
            def _run_delegate(self, prompt: str) -> str | None:  # type: ignore[override]
                seen["prompt"] = prompt
                path = self._project_root / "billing_agent.py"
                path.write_text(
                    path.read_text(encoding="utf-8").replace(
                        '"We have plans."', '"Premium is $9.99 per month."'
                    ),
                    encoding="utf-8",
                )
                return None

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _Capturing(git_project, runner, "true")
        delegate.attempt(compile_goals(parse_goals(GOALS)), _failing_cases(git_project))

        assert "LESSONS FROM PAST ACCEPTED FIXES" in seen["prompt"]
        assert "Prefer $9.99 pricing" in seen["prompt"]

    def test_distils_an_accepted_fix_into_playbook(self, git_project: Path) -> None:
        from agentopt.improver.code_only import CodeOnlyLoop
        from agentopt.improver.reflector import Reflector
        from agentopt.paths import Workspace

        class _FakeBackend:
            def complete(self, messages: object, **_: object) -> str:
                return "Set the premium price to $9.99 in the pricing constant"

        command = _fake_delegate(git_project, FIXING_DELEGATE)
        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        delegate = _SubprocessDelegate(git_project, runner, command)

        outcome = CodeOnlyLoop(
            project_root=git_project,
            runner=runner,
            judge=None,
            repeats=1,
            tier4=delegate,
            ledger_path=Workspace(git_project).ledger_file,
            max_rounds=1,
            max_changes=5,
            plateau_stop=2,
            reflector=Reflector(_FakeBackend()),  # type: ignore[arg-type]
        ).run(compile_goals(parse_goals(GOALS)))

        assert len(outcome.accepted) == 1
        playbook = git_project / "playbook.md"
        assert playbook.is_file()
        assert "Set the premium price to $9.99" in playbook.read_text(encoding="utf-8")

    def test_narrates_baseline_playbook_scoring_and_lesson(self, git_project: Path) -> None:
        # The run should SAY what context it is using: baseline, playbook memory,
        # scoring progress, and the lesson it learns.
        from agentopt.improver.code_only import CodeOnlyLoop
        from agentopt.improver.reflector import Reflector
        from agentopt.paths import Workspace
        from agentopt.progress import set_progress_sink

        (git_project / "playbook.md").write_text(
            "# Playbook\n\n- Prefer $9.99 pricing <!-- id:abc -->\n", encoding="utf-8"
        )

        class _FakeBackend:
            def complete(self, messages: object, **_: object) -> str:
                return "Set the premium price to $9.99"

        lines: list[str] = []
        set_progress_sink(lines.append)
        try:
            command = _fake_delegate(git_project, FIXING_DELEGATE)
            runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
            delegate = _SubprocessDelegate(git_project, runner, command)
            CodeOnlyLoop(
                project_root=git_project,
                runner=runner,
                judge=None,
                repeats=1,
                tier4=delegate,
                ledger_path=Workspace(git_project).ledger_file,
                max_rounds=1,
                max_changes=5,
                plateau_stop=2,
                reflector=Reflector(_FakeBackend()),  # type: ignore[arg-type]
            ).run(compile_goals(parse_goals(GOALS)))
        finally:
            set_progress_sink(None)

        log = "\n".join(lines)
        assert "preflight: checking the agent can run" in log
        assert "measuring baseline:" in log
        assert "using 1 playbook lesson(s) from past accepted fixes" in log
        assert re.search(r"scoring: \d+/\d+ runs", log)
        assert re.search(r"judging: \d+/\d+ cases", log)
        assert "playbook lesson learned: Set the premium price to $9.99" in log

    def test_feeds_a_rejected_approach_into_the_retry_prompt(self, git_project: Path) -> None:
        # #3: the first attempt makes a non-improving edit (reverted); the retry
        # must be TOLD that dead path so it doesn't blindly re-roll into it.
        from agentopt.improver.code_only import CodeOnlyLoop
        from agentopt.paths import Workspace

        prompts: list[str] = []
        calls = {"n": 0}

        class _TwoShot(_SubprocessDelegate):
            def _run_delegate(self, prompt: str) -> str | None:  # type: ignore[override]
                prompts.append(prompt)
                calls["n"] += 1
                path = self._project_root / "billing_agent.py"
                # First: a wrong edit (no gain → reverted). Retry: the real fix.
                good = '"Premium is $9.99 per month."'
                replacement = '"Plans start at $1.11."' if calls["n"] == 1 else good
                path.write_text(
                    path.read_text(encoding="utf-8").replace('"We have plans."', replacement),
                    encoding="utf-8",
                )
                return None

        runner = runner_for(load_config(git_project / "agentopt.yaml"), git_project)
        outcome = CodeOnlyLoop(
            project_root=git_project,
            runner=runner,
            judge=None,
            repeats=1,
            tier4=_TwoShot(git_project, runner, "true"),
            ledger_path=Workspace(git_project).ledger_file,
            max_rounds=1,
            max_changes=5,
            plateau_stop=2,
            max_attempts_per_cluster=2,
        ).run(compile_goals(parse_goals(GOALS)))

        assert calls["n"] == 2  # retried after the first was reverted
        assert len(outcome.accepted) == 1
        # The retry's prompt carries the negative memory of the reverted first try.
        assert "ALREADY TRIED AND REVERTED" in prompts[1]
        assert "$1.11" in prompts[1]  # the exact dead-path diff, so it won't repeat it
        assert "ALREADY TRIED AND REVERTED" not in prompts[0]  # nothing to warn about yet


class TestEnvironmentGuardAndUsage:
    """B4 (abort on broken environment) + B5 (black-box token metering)."""

    def test_agent_runs_mostly_failed_flags_a_broken_environment(self) -> None:
        from datetime import datetime, timezone

        from agentopt.evals.scorecard import Scorecard, agent_runs_mostly_failed

        def card(total: int, failed: int) -> Scorecard:
            return Scorecard(
                created_at=datetime.now(timezone.utc),
                score=0,
                measured_cases=0,
                skipped_cases=0,
                runs_total=total,
                runs_failed=failed,
            )

        assert agent_runs_mostly_failed(card(10, 6)) is True  # 60% → environment problem
        assert agent_runs_mostly_failed(card(10, 4)) is False  # 40% → still optimizable
        assert agent_runs_mostly_failed(card(0, 0)) is False  # nothing ran → not a signal

    def test_meters_black_box_tokens_from_self_reported_usage(self) -> None:
        from agentopt.improver.budget import MeteredRunner
        from agentopt.runner import AgentRunner, RunResult
        from agentopt.wiretap.context import Mode
        from agentopt.wiretap.overlay import Overlay

        class _Fake(AgentRunner):
            def __init__(self) -> None:
                super().__init__(Path("."))

            def run(
                self,
                input_text: str | None,
                mode: Mode,
                overlay: Overlay | None = None,
                surface: str | None = None,
            ) -> RunResult:
                # No trace in black-box mode — the count must come from usage.
                return RunResult(
                    run_id="r",
                    trace_path=Path("does-not-exist"),
                    output="ok",
                    exit_code=0,
                    usage={"input": 400, "output": 100},
                )

        metered = MeteredRunner(_Fake(), 10_000)
        metered.run("hi", mode="eval")
        assert metered.tokens_used == 500


class TestDelegateStreaming:
    """#1: the delegate output is streamed with a bounded tail, not fully buffered."""

    def test_returns_tail_on_failure_without_buffering(self, tmp_path: Path) -> None:
        # A fake 'claude' that floods stdout then exits non-zero. We must get the
        # TAIL as the error (not the whole flood).
        script = tmp_path / "flood.py"
        script.write_text(
            "import sys\n"
            "for i in range(5000):\n    print('line', i, 'padding padding padding')\n"
            "print('BOOM_THE_REAL_ERROR')\n"
            "sys.exit(3)\n",
            encoding="utf-8",
        )
        # ClaudeCliAgent runs `<exe> -p <prompt> ...`; a tiny shim executable runs
        # the flood script and ignores those args.
        shim = tmp_path / ("shim.cmd" if sys.platform == "win32" else "shim.sh")
        if sys.platform == "win32":
            shim.write_text(f'@echo off\r\n"{sys.executable}" "{script}"\r\n', encoding="utf-8")
        else:
            shim.write_text(f'#!/bin/sh\n"{sys.executable}" "{script}"\n', encoding="utf-8")
            shim.chmod(0o755)
        err = ClaudeCliAgent(str(shim)).run(tmp_path, "the prompt", 30)
        assert err is not None
        assert "exited 3" in err
        assert "BOOM_THE_REAL_ERROR" in err  # tail captured
        assert len(err) < 400  # only the tail, not 5000 lines

    def test_returns_none_on_success(self, tmp_path: Path) -> None:
        shim = tmp_path / ("ok.cmd" if sys.platform == "win32" else "ok.sh")
        if sys.platform == "win32":
            shim.write_text("@echo off\r\necho done\r\nexit /b 0\r\n", encoding="utf-8")
        else:
            shim.write_text("#!/bin/sh\necho done\nexit 0\n", encoding="utf-8")
            shim.chmod(0o755)
        assert ClaudeCliAgent(str(shim)).run(tmp_path, "p", 30) is None


class TestCodeAgentSelection:
    def test_claude_backend_uses_the_cli_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agentopt.improver.tier4.shutil.which", lambda _: "/usr/bin/claude")
        agent, detail = build_code_agent(ImproverConfig(backend="claude-code"))
        assert isinstance(agent, ClaudeCliAgent)
        assert "Claude Code CLI" in detail

    def test_claude_backend_without_cli_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agentopt.improver.tier4.shutil.which", lambda _: None)
        agent, detail = build_code_agent(ImproverConfig(backend="claude-code"))
        assert agent is None
        assert "not found" in detail

    def test_codex_backend_uses_the_codex_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("agentopt.improver.tier4.shutil.which", lambda _: "/usr/bin/codex")
        agent, detail = build_code_agent(ImproverConfig(backend="codex"))
        assert isinstance(agent, CodexCliAgent)
        assert "Codex CLI" in detail

    def test_explicit_codex_delegate_overrides_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # delegate: codex runs the CODE agent on Codex regardless of backend.
        monkeypatch.setattr("agentopt.improver.tier4.shutil.which", lambda _: "/usr/bin/codex")
        agent, detail = build_code_agent(ImproverConfig(backend="cloud", delegate="codex"))
        assert isinstance(agent, CodexCliAgent)
        assert "Codex CLI" in detail

    def test_explicit_claude_delegate_overrides_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agentopt.improver.tier4.shutil.which", lambda _: "/usr/bin/claude")
        agent, detail = build_code_agent(ImproverConfig(backend="cloud", delegate="claude-code"))
        assert isinstance(agent, ClaudeCliAgent)
        assert "Claude Code CLI" in detail

    def test_codex_delegate_without_cli_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agentopt.improver.tier4.shutil.which", lambda _: None)
        agent, detail = build_code_agent(ImproverConfig(backend="codex"))
        assert agent is None
        assert "not found" in detail

    def test_cloud_backend_uses_deepagents_with_the_yaml_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("agentopt.improver.tier4.shutil.which", lambda _: None)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        agent, detail = build_code_agent(
            ImproverConfig(backend="cloud", optimizer="claude-sonnet-4-5")
        )
        assert isinstance(agent, DeepAgentsAgent)
        assert detail == "deepagents (anthropic:claude-sonnet-4-5)"

    def test_provider_prefixed_optimizer_is_used_verbatim(self) -> None:
        agent, detail = build_code_agent(
            ImproverConfig(backend="cloud", optimizer="openai:gpt-5.5")
        )
        assert isinstance(agent, DeepAgentsAgent)
        assert detail == "deepagents (openai:gpt-5.5)"

    def test_ollama_backend_defaults_a_local_model(self) -> None:
        agent, detail = build_code_agent(ImproverConfig(backend="ollama"))
        assert isinstance(agent, DeepAgentsAgent)
        assert detail == "deepagents (ollama:llama3.1)"

    def test_cloud_without_model_or_key_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        agent, detail = build_code_agent(ImproverConfig(backend="cloud"))
        assert agent is None
        assert "improver.optimizer" in detail

    def test_deepagents_run_invokes_the_agent_with_the_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents

        seen: dict[str, object] = {}

        class _FakeAgent:
            def invoke(self, payload: dict, config: dict) -> None:  # type: ignore[type-arg]
                seen["prompt"] = payload["messages"][0]["content"]
                seen["recursion"] = config["recursion_limit"]

        def _fake_create(*, model: str, backend: object, system_prompt: str) -> _FakeAgent:
            seen["model"] = model
            seen["backend"] = type(backend).__name__
            return _FakeAgent()

        monkeypatch.setattr(deepagents, "create_deep_agent", _fake_create)
        error = DeepAgentsAgent("anthropic:claude-sonnet-4-5").run(tmp_path, "fix it", timeout=10)

        assert error is None
        assert seen["model"] == "anthropic:claude-sonnet-4-5"
        assert seen["prompt"] == "fix it"
        assert seen["backend"] == "FilesystemBackend"  # real backend, project-scoped

    def test_deepagents_surfaces_a_delegate_failure_as_a_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import deepagents

        def _boom(**_kwargs: object) -> object:
            raise RuntimeError("model unreachable")

        monkeypatch.setattr(deepagents, "create_deep_agent", _boom)
        error = DeepAgentsAgent("anthropic:x").run(tmp_path, "fix it", timeout=10)

        assert error is not None
        assert "deepagents delegate failed" in error
        assert "model unreachable" in error

    def test_delegate_with_no_agent_is_refused(self, git_project: Path) -> None:
        config = load_config(git_project / "agentopt.yaml")
        runner = runner_for(config, git_project)
        delegate = CodeDelegate(
            git_project,
            runner,
            judge=None,
            repeats=1,
            code_agent=None,
            unavailable_reason="tier-4: nothing available",
        )
        result = delegate.attempt(compile_goals(parse_goals(GOALS)), [])
        assert not result.attempted
        assert result.reason == "tier-4: nothing available"


class TestWebsearch:
    def test_non_claude_backend_reports_unavailable(self) -> None:
        class _Other:
            name = "other"

            def complete(self, messages: list[ChatMessage], **kwargs: object) -> str:
                return "should not be called"

        assert search_guidance(_Other(), "problem") is None  # type: ignore[arg-type]

    def test_claude_backend_failure_is_fail_soft(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agentopt.gateway.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend(executable="claude")
        monkeypatch.setattr(
            ClaudeCodeBackend,
            "complete",
            lambda self, messages, **kwargs: (_ for _ in ()).throw(GatewayError("down")),
        )
        assert search_guidance(backend, "problem") is None

    def test_grounded_prompt_injects_model_and_source_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agentopt.gateway.claude_code import ClaudeCodeBackend

        seen: list[ChatMessage] = []

        def fake_complete(self: object, messages: list[ChatMessage], **_: object) -> str:
            seen.extend(messages)
            return "SEARCHED: use an output-format constraint - ... - https://docs.example"

        monkeypatch.setattr(ClaudeCodeBackend, "complete", fake_complete)
        out = search_guidance(
            ClaudeCodeBackend(executable="claude"),
            "GOAL(S) FAILING:\n- be concise",
            model="kimi-k2p6",
        )
        assert out is not None and out.startswith("SEARCHED:")
        sent = "\n".join(m.content for m in seen)
        assert "OPEN and READ" in sent  # instructs it to actually fetch sources
        assert "SOURCE_URL" in sent  # requires citations
        assert 'model "kimi-k2p6"' in sent  # provider-aware
