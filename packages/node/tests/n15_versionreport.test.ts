/** N15: the version-report renderer — multi-file diffs for code (tier-4) edits. */

import { describe, expect, it } from "vitest";
import { renderVersionReport, unifiedDiff } from "../src/improver/versionreport.js";
import { buildDecision } from "../src/improver/decisionreport.js";
import type { CaseResult, Scorecard } from "../src/evals/engine.js";

function dcase(id: string, behavior: number | null, valid: number, n: number, k: number): CaseResult {
  return {
    case_id: id, source: id, layer: 2, runs: n, passed_runs: k,
    score: n ? k / n : 0, status: k > n / 2 ? "passed" : "failed",
    detail: "", valid_runs: valid, behavior_score: behavior,
  };
}
function dcard(cases: CaseResult[], reliability: number): Scorecard {
  return {
    created_at: "2026-01-01T00:00:00Z",
    score: (cases.reduce((s, c) => s + c.score, 0) / cases.length) * 100,
    measured_cases: cases.length, skipped_cases: 0, cases, warnings: [],
    behavior_quality: (cases.reduce((s, c) => s + (c.behavior_score ?? 0), 0) / cases.length) * 100,
    execution_reliability: reliability,
  };
}

describe("version report renderer", () => {
  it("renders a single-file prompt/param edit without a files-changed line", () => {
    const md = renderVersionReport({
      version: "v3",
      title: "raise max_tokens to 4096",
      targetId: "router_system_prompt",
      targetFile: "src/llm.ts",
      evidence: "72% -> 99%",
      appliedAt: "2026-08-11T00:00:00Z",
      files: [{ path: "src/llm.ts", before: "max_tokens: 5", after: "max_tokens: 4096" }],
    });
    expect(md).toContain("# v3 — raise max_tokens to 4096");
    expect(md).toContain("- target: router_system_prompt (src/llm.ts)");
    expect(md).not.toContain("files changed:"); // single file → no list
    expect(md).toContain("```diff");
    expect(md).toContain("-max_tokens: 5");
    expect(md).toContain("+max_tokens: 4096");
  });

  it("renders every file a code edit rewrote, each with its own diff", () => {
    const md = renderVersionReport({
      version: "v4",
      title: "delegated code fix (src/tools.ts, src/router.ts)",
      targetId: "workflow",
      targetFile: null,
      evidence: "train 60% -> 100%, no regressions",
      appliedAt: "2026-08-11T00:00:00Z",
      files: [
        { path: "src/tools.ts", before: "desc: 'old'", after: "desc: 'clear, specific'" },
        { path: "src/router.ts", before: "return 'chat'", after: "return classify(input)" },
      ],
    });
    expect(md).toContain("- files changed: src/tools.ts, src/router.ts");
    // Both files appear as diff headers with their own hunks.
    expect(md).toContain("--- src/tools.ts (before)");
    expect(md).toContain("--- src/router.ts (before)");
    expect(md).toContain("-desc: 'old'");
    expect(md).toContain("+return classify(input)");
    expect((md.match(/```diff/g) ?? []).length).toBe(2); // one fenced block per file
  });

  it("skips a file with no pre-image (delegate-created) but keeps the rest", () => {
    const md = renderVersionReport({
      version: "v5",
      title: "delegated code fix",
      targetId: "workflow",
      targetFile: null,
      evidence: "ok",
      appliedAt: "2026-08-11T00:00:00Z",
      files: [
        { path: "src/new.ts", before: null, after: "brand new" }, // created → skipped in diff
        { path: "src/edited.ts", before: "a", after: "b" },
      ],
    });
    expect((md.match(/```diff/g) ?? []).length).toBe(1); // only the edited file diffs
    expect(md).toContain("--- src/edited.ts (before)");
    expect(md).not.toContain("--- src/new.ts (before)");
  });

  it("surfaces the gate verdict prominently and labels a dipped headline as sampled", () => {
    const md = renderVersionReport({
      version: "v7",
      title: "Add concrete examples for split changes",
      targetId: "async_routeintent_system_prompt",
      targetFile: "src/agent.ts",
      evidence: "train 78% -> 81%, holdout 83% -> 83%, 2 failing case(s) fixed, no regressions",
      sampledBefore: 80.04,
      sampledAfter: 79.94, // fractionally lower — sampling noise, not a regression
      appliedAt: "2026-08-11T00:00:00Z",
      files: [{ path: "src/agent.ts", before: "a", after: "b" }],
    });
    // The verdict leads the report, not a buried bullet.
    expect(md).toContain("> **Why it was accepted:** train 78% -> 81%, holdout 83% -> 83%");
    // The noisy headline is shown but explicitly labelled sampled.
    expect(md).toContain("- overall suite score (sampled): 80.0% → 79.9%");
    // A clarifier explains the dip so it doesn't read as a regression.
    expect(md).toContain("sampling noise, not a");
    expect(md).toContain("no case made significantly worse");
  });

  it("omits the sampled line and note when no sampled scores are given", () => {
    const md = renderVersionReport({
      version: "v1",
      title: "x",
      targetId: "t",
      targetFile: null,
      evidence: "ok",
      appliedAt: "2026-08-11T00:00:00Z",
      files: [{ path: "a.ts", before: "a", after: "b" }],
    });
    expect(md).toContain("> **Why it was accepted:** ok");
    expect(md).not.toContain("overall suite score (sampled)");
    expect(md).not.toContain("sampling noise");
  });

  it("unifiedDiff returns just the headers when nothing changed", () => {
    expect(unifiedDiff("same", "same", "a", "b")).toEqual(["--- a", "+++ b"]);
  });
});

describe("buildVersionReport (machine-readable source of truth)", () => {
  it("produces the structured report with candidate, diff, and rollback command", async () => {
    const { buildVersionReport } = await import("../src/improver/versionreport.js");
    const report = buildVersionReport({
      version: "v5",
      title: "code-only delegated fix",
      tier: 4,
      targetId: "workflow",
      evidence: "train 80% -> 92%, no regressions",
      appliedAt: "2026-08-15T10:00:00Z",
      files: [{ path: "agent.ts", before: "a = 1\n", after: "a = 2\n" }],
      rollbackCommand: "agentopt rollback v5",
    });
    // JSON-serializable (persisted as {version}.json) with the essentials.
    const round = JSON.parse(JSON.stringify(report));
    expect(round.accepted).toBe(true);
    expect(round.version).toBe("v5");
    expect(round.rollback_command).toBe("agentopt rollback v5");
    expect(round.candidate).toEqual({ description: "code-only delegated fix", tier: 4, target_id: "workflow" });
    expect(round.decision_ref).toBeNull();
    expect(round.changes.files[0].diff).toContain("+a = 2");
  });

  it("carries a decision_ref when one is supplied", async () => {
    const { buildVersionReport } = await import("../src/improver/versionreport.js");
    const report = buildVersionReport({
      version: "v6", title: "fix", tier: 4, targetId: "code",
      evidence: "ok", appliedAt: "2026-08-15T10:00:00Z",
      files: [], rollbackCommand: "agentopt rollback v6",
      decisionRef: "decisions/v6.json",
    });
    expect((report as { decision_ref: string }).decision_ref).toBe("decisions/v6.json");
  });
});

describe("renderVersionHtml (shareable before/after view)", () => {
  it("is self-contained and shows the verdict, evidence, and colorized diff", async () => {
    const { buildVersionReport, renderVersionHtml } = await import("../src/improver/versionreport.js");
    const report = buildVersionReport({
      version: "v7", title: "raise max_tokens", tier: 4, targetId: "workflow",
      evidence: "train 80% -> 92%, no regressions", appliedAt: "2026-08-15T10:00:00Z",
      files: [{ path: "agent.ts", before: "max_tokens = 5\n", after: "max_tokens = 4096\n" }],
      rollbackCommand: "agentopt rollback v7", decisionRef: "decisions/v7.json",
    });
    const html = renderVersionHtml(report);
    expect(html.startsWith("<!doctype html>")).toBe(true);
    expect(html).not.toContain("http://");
    expect(html).not.toContain("https://");
    expect(html).not.toContain('src="');
    expect(html).toContain("accepted");
    expect(html).toContain("train 80% -&gt; 92%"); // evidence HTML-escaped
    expect(html).toContain('class="add"');
    expect(html).toContain("max_tokens = 4096");
    expect(html).toContain("agentopt rollback v7");
    expect(html).toContain("decisions/v7.json");
  });

  it("escapes diff content instead of injecting it", async () => {
    const { buildVersionReport, renderVersionHtml } = await import("../src/improver/versionreport.js");
    const report = buildVersionReport({
      version: "v8", title: "fix", tier: 4, targetId: "workflow",
      evidence: "ok", appliedAt: "2026-08-15T10:00:00Z",
      files: [{ path: "a.ts", before: "x = 1\n", after: "x = '<script>'\n" }],
      rollbackCommand: "agentopt rollback v8",
    });
    const html = renderVersionHtml(report);
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
  });
});

describe("buildDecision (structured gate verdict)", () => {
  const meta = { id: "v3", tier: 4, target_id: "code", description: "a fix" };
  const baseline = dcard([dcase("a", 0.5, 8, 8, 4)], 1.0);

  it("emits an accepted decision with the lexicographic stages passed", () => {
    const good = dcard([dcase("a", 1.0, 8, 8, 8)], 1.0);
    const decision = buildDecision({
      candidate: meta, baseline, candidateCard: good,
      holdoutBefore: null, holdoutAfter: null, createdAt: "2026-01-01T00:00:00Z",
    }) as { accepted: boolean; stages: Array<{ status: string }>; rejection: unknown };
    expect(decision.accepted).toBe(true);
    expect(decision.stages.slice(0, 4).map((s) => s.status)).toEqual(["passed", "passed", "passed", "passed"]);
    expect(decision.rejection).toBeNull();
  });

  it("attributes a reliability rejection to the reliability stage, skipping later ones", () => {
    const flaky = dcard([dcase("a", 1.0, 8, 8, 8)], 0.8);
    const decision = buildDecision({
      candidate: meta, baseline, candidateCard: flaky,
      holdoutBefore: null, holdoutAfter: null, createdAt: "2026-01-01T00:00:00Z",
    }) as {
      accepted: boolean;
      rejection: { stage: string };
      stages: Array<{ stage: string; status: string }>;
    };
    expect(decision.accepted).toBe(false);
    expect(decision.rejection.stage).toBe("reliability");
    const byStage = Object.fromEntries(decision.stages.map((s) => [s.stage, s.status]));
    expect(byStage.reliability).toBe("failed");
    expect(byStage.quality).toBe("skipped");
  });

  it("persistDecision writes evidence scorecards and links them", async () => {
    const { persistDecision } = await import("../src/improver/decisionreport.js");
    const { mkdtempSync, existsSync } = await import("node:fs");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const dir = mkdtempSync(join(tmpdir(), "agentopt-dec-"));
    const good = dcard([dcase("a", 1.0, 8, 8, 8)], 1.0);
    const { decisionRef, decision } = persistDecision(dir, "v4", {
      candidate: meta, baseline, candidateCard: good,
      holdoutBefore: null, holdoutAfter: null, createdAt: "2026-01-01T00:00:00Z",
      criticalIds: null, caseWeights: null, verdict: { accepted: true, reason: "ok" },
    });
    expect(decisionRef).toBe("decisions/v4.json");
    expect(existsSync(join(dir, "decisions", "v4.json"))).toBe(true);
    expect(existsSync(join(dir, "evidence", "v4", "baseline_train.json"))).toBe(true);
    expect(existsSync(join(dir, "evidence", "v4", "candidate_train.json"))).toBe(true);
    const refs = (decision as { evidence_refs: Record<string, string | null> }).evidence_refs;
    expect(refs.baseline_train).toBe("evidence/v4/baseline_train.json");
    expect(refs.baseline_holdout).toBeNull();
  });

  it("forces the artifact verdict to match the gate's authoritative decision", () => {
    // Re-derivation would REJECT (trivial quality dip below floor), but the gate
    // ACCEPTED (e.g. its resampling passed) — the artifact must match the gate.
    const worse = dcard([dcase("a", 0.895, 8, 8, 8)], 1.0);
    const base = dcard([dcase("a", 0.9, 8, 8, 8)], 1.0);
    const decision = buildDecision({
      candidate: meta, baseline: base, candidateCard: worse,
      holdoutBefore: null, holdoutAfter: null, createdAt: "2026-01-01T00:00:00Z",
      verdict: { accepted: true, reason: "resampled to noise, no regressions" },
    }) as { accepted: boolean; rejection: unknown };
    expect(decision.accepted).toBe(true);
    expect(decision.rejection).toBeNull();
  });

  it("attributes a rejection to the gate's REAL stage, not a guess", () => {
    // The derivation would ACCEPT (real gain, reliability held), but the gate rejected
    // at RELIABILITY — the artifact must name that stage, not "quality".
    const good = dcard([dcase("a", 1.0, 8, 8, 8)], 1.0);
    const decision = buildDecision({
      candidate: meta, baseline, candidateCard: good,
      holdoutBefore: null, holdoutAfter: null, createdAt: "2026-01-01T00:00:00Z",
      verdict: { accepted: false, reason: "execution reliability regressed", stage: "reliability" },
    }) as { accepted: boolean; rejection: { stage: string } };
    expect(decision.accepted).toBe(false);
    expect(decision.rejection.stage).toBe("reliability");
  });

  // --- auditable cost/latency tiebreak (#4) ---
  type Dec = {
    accepted: boolean;
    stages: Array<{ stage: string; status: string; detail?: string }>;
    dimensions: {
      operational_cost: {
        preference: string;
        basis: string;
        candidate_tokens: number | null;
      };
    };
  };
  const decide = (base: Scorecard, cand: Scorecard, verdict: Parameters<typeof buildDecision>[0]["verdict"]) =>
    buildDecision({
      candidate: meta, baseline: base, candidateCard: cand,
      holdoutBefore: null, holdoutAfter: null, createdAt: "2026-01-01T00:00:00Z", verdict,
    }) as Dec;

  it("tiebreak prefers fewer tokens when quality is equal", () => {
    const base = { ...dcard([dcase("a", 1.0, 8, 8, 8)], 1.0), total_tokens: 1200, total_latency_ms: 4200 };
    const cand = { ...dcard([dcase("a", 1.0, 8, 8, 8)], 1.0), total_tokens: 900, total_latency_ms: 3900 };
    const d = decide(base, cand, { accepted: true, reason: "ok" });
    expect(d.dimensions.operational_cost.basis).toBe("tokens");
    expect(d.dimensions.operational_cost.preference).toBe("candidate");
    const tb = d.stages.find((s) => s.stage === "tiebreak")!;
    expect(tb.detail).toContain("tokens");
    expect(tb.detail).toContain("900");
  });

  it("tiebreak falls to latency when tokens are tied", () => {
    const base = { ...dcard([dcase("a", 1.0, 8, 8, 8)], 1.0), total_tokens: 1000, total_latency_ms: 5000 };
    const cand = { ...dcard([dcase("a", 1.0, 8, 8, 8)], 1.0), total_tokens: 1000, total_latency_ms: 3000 };
    const oc = decide(base, cand, { accepted: true, reason: "ok" }).dimensions.operational_cost;
    expect(oc.basis).toBe("latency");
    expect(oc.preference).toBe("candidate");
  });

  it("tiebreak preference is unknown with no cost data", () => {
    const base = dcard([dcase("a", 1.0, 8, 8, 8)], 1.0);
    const cand = dcard([dcase("a", 1.0, 8, 8, 8)], 1.0);
    const oc = decide(base, cand, { accepted: true, reason: "ok" }).dimensions.operational_cost;
    expect(oc.preference).toBe("unknown");
    expect(oc.basis).toBe("none");
  });

  it("a cheaper candidate never overrides a failed earlier stage", () => {
    const base = { ...dcard([dcase("a", 0.5, 8, 8, 4)], 1.0), total_tokens: 9999 };
    const cand = { ...dcard([dcase("a", 1.0, 8, 8, 8)], 0.5), total_tokens: 10 };
    const d = decide(base, cand, { accepted: false, reason: "execution reliability regressed", stage: "reliability" });
    expect(d.accepted).toBe(false);
    const byStage = Object.fromEntries(d.stages.map((s) => [s.stage, s.status]));
    expect(byStage.reliability).toBe("failed");
    expect(byStage.tiebreak).toBe("skipped");
  });
});
