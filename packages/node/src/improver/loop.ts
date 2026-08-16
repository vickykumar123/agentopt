/**
 * The improvement loop: measure → cluster → reflect → trial → verify →
 * apply, over multiple rounds until plateau.
 *
 * Each round re-measures (applied changes shift the baseline), attributes
 * the remaining failures to editable prompts via their traces, and fixes
 * causes — one candidate per failure cluster. The ledger remembers every
 * attempt across rounds and runs; clusters that stay stuck become help-me
 * cards; accepted changes distill into playbook lessons.
 */

import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { abortMessage, agentRunsMostlyFailed, EvalEngine, preflight, PASS_THRESHOLD, singleCasePlan, type CaseResult, type Judge, type RunData, type Scorecard } from "../evals/engine.js";
import {
  applyHistory,
  flakyCaseIds,
  loadObservations,
  observationsPathFor,
  recordScores,
  saveObservations,
} from "./observations.js";
import { calibrateSuite, unreliableRubrics } from "../evals/calibrate.js";
import { establishCriticalBaseline, RESAMPLE_STEP } from "./scoregain.js";
import { findRotationSalt, persistSuite, splitSuite, suitesEqual } from "../evals/suite.js";
import type { LlmBackend } from "../gateway.js";
import type { EvalCase, EvalSuite } from "../goals.js";
import { PromptLocator } from "../manifest/locator.js";
import { attemptsForTarget, loadLedger, saveLedger, type Attempt, type Ledger } from "../models/ledger.js";
import { ManifestSchema, type Manifest, type ManifestEntry } from "../models/manifest.js";
import { Workspace } from "../paths.js";
import { PlaybookStore, relevantBullets } from "../playbook.js";
import type { AgentRunner } from "../runner.js";
import type { Overlay } from "../wiretap/overlay.js";
import type { SafeWriter } from "../writer/safeWriter.js";
import { WriterError } from "../writer/safeWriter.js";
import { clusterFailures, llmCallsForFailure, type FailureCluster } from "./attribution.js";
import { BudgetExceededError } from "./budget.js";
import { candidateFingerprint, overlayFor, type Candidate, type EntryContent } from "./candidates.js";
import { VerificationGate, type GateResult } from "./gate.js";
import type { CodeDelegate } from "./tier4.js";
import { writeHelpCards, type HelpCard } from "./helpcards.js";
import { Reflector } from "./reflector.js";
import { buildVersionReport, renderVersionHtml, renderVersionReport, unifiedDiff, type ReportFile } from "./versionreport.js";
import { persistDecision } from "./decisionreport.js";

/** Live progress line — a long run must never feel hung. */
function say(message: string): void {
  console.log(message);
}

export const DEFAULT_MAX_CANDIDATES = 3;
/** Rounds a cluster may fail before it becomes a help-me card. */
const STUCK_THRESHOLD = 2;

export interface VerifiedChange {
  candidate: Candidate;
  overlay: Overlay;
  result: GateResult;
  entry: ManifestEntry;
  version: string | null;
  /** Files a code (tier-4) change rewrote; unset for single-file prompt/param edits. */
  changed_files?: string[];
}

export interface ImproveOutcome {
  baseline: Scorecard;
  finalScore: number;
  roundsRun: number;
  accepted: VerifiedChange[];
  rejected: GateResult[];
  notes: string[];
  lessonsAdded: string[];
  helpCardsPath: string | null;
}

export interface ImproveLoopOptions {
  projectRoot: string;
  runner: AgentRunner;
  backend: LlmBackend;
  optimizerModel: string | null;
  judge: Judge | null;
  repeats: number;
  ledgerPath: string;
  maxCandidates?: number;
  maxRounds?: number;
  plateauStop?: number;
  strictHoldout?: boolean;
  maxChanges?: number;
  /** Diverse candidates generated + probed per cluster each round (exploration). */
  explore?: number;
  writer?: SafeWriter | null;
  searcher?: ((problem: string) => Promise<string | null>) | null;
  challenger?: ((suite: EvalSuite) => Promise<EvalCase[]>) | null;
  tier4?: CodeDelegate | null;
  /** Agent runs to score concurrently (bounded pool). 1 = serial. */
  concurrency?: number;
  /** The configured min-effect / reliability thresholds — passed to the overlay gate
   * so it holds candidates to the SAME standard as the code path. */
  minEffectPP?: number;
  reliabilityTolerance?: number;
}

export class ImproveLoop {
  private readonly projectRoot: string;
  private readonly workspace: Workspace;
  private readonly runner: AgentRunner;
  private readonly reflector: Reflector;
  private readonly gate: VerificationGate;
  private readonly judge: Judge | null;
  private readonly repeats: number;
  private readonly ledgerPath: string;
  private readonly observationsPath: string;
  private paramsHintFile: string | null = null;
  private readonly maxCandidates: number;
  private readonly maxRounds: number;
  private readonly plateauStop: number;
  private readonly maxChanges: number;
  private readonly explore: number;
  private readonly writer: SafeWriter | null;
  private readonly searcher: ((problem: string) => Promise<string | null>) | null;
  /** Per-run memo of web guidance, keyed by failing-goal set (search is costly). */
  private readonly searchCache = new Map<string, string | null>();
  private readonly challenger: ((suite: EvalSuite) => Promise<EvalCase[]>) | null;
  private readonly tier4: CodeDelegate | null;
  private readonly concurrency: number;
  private challengesAdded = false;
  private locator: PromptLocator;
  private readonly playbook: PlaybookStore;
  private readonly stuckRounds = new Map<string, number>();
  private readonly clusterAttempts = new Map<string, string[]>();
  private partialOutcome: ImproveOutcome | null = null;

  constructor(options: ImproveLoopOptions) {
    this.projectRoot = options.projectRoot;
    this.workspace = new Workspace(options.projectRoot);
    this.runner = options.runner;
    this.reflector = new Reflector(options.backend, options.optimizerModel);
    this.concurrency = Math.max(1, options.concurrency ?? 1);
    this.gate = new VerificationGate(
      options.runner,
      options.judge,
      options.repeats,
      options.strictHoldout ?? false,
      this.concurrency,
      options.minEffectPP,
      options.reliabilityTolerance,
    );
    this.judge = options.judge;
    this.repeats = options.repeats;
    this.ledgerPath = options.ledgerPath;
    this.observationsPath = observationsPathFor(options.ledgerPath);
    this.maxCandidates = options.maxCandidates ?? DEFAULT_MAX_CANDIDATES;
    this.maxRounds = Math.max(1, options.maxRounds ?? 5);
    this.plateauStop = Math.max(1, options.plateauStop ?? 2);
    this.maxChanges = Math.max(1, options.maxChanges ?? 5);
    this.explore = Math.max(1, options.explore ?? 3);
    this.writer = options.writer ?? null;
    this.searcher = options.searcher ?? null;
    this.challenger = options.challenger ?? null;
    this.tier4 = options.tier4 ?? null;
    this.locator = new PromptLocator(options.projectRoot);
    this.playbook = new PlaybookStore(options.projectRoot, this.workspace.root);
  }

  async run(suite: EvalSuite, manifest: Manifest): Promise<ImproveOutcome> {
    try {
      return await this.runInner(suite, manifest);
    } catch (error) {
      if (!(error instanceof BudgetExceededError)) throw error;
      const outcome: ImproveOutcome = this.partialOutcome ?? {
        baseline: emptyScorecard(),
        finalScore: 0,
        roundsRun: 0,
        accepted: [],
        rejected: [],
        notes: [],
        lessonsAdded: [],
        helpCardsPath: null,
      };
      outcome.notes.push(`stopping: ${error.message} — applied versions are kept`);
      saveLedger(loadLedger(this.ledgerPath), this.ledgerPath);
      return outcome;
    }
  }

  /** Early-return outcome that changes nothing — used when a run aborts. */
  private abortOutcome(baseline: Scorecard, note: string): ImproveOutcome {
    console.log(note);
    return {
      baseline,
      finalScore: baseline.score ?? 0,
      roundsRun: 0,
      accepted: [],
      rejected: [],
      notes: [note],
      lessonsAdded: [],
      helpCardsPath: null,
    };
  }

  private async runInner(suite: EvalSuite, manifest: Manifest): Promise<ImproveOutcome> {
    let salt = 0;
    // Where the agent's model call lives — the hint for applying a params change.
    this.paramsHintFile = manifest.entries.find((e) => e.kind === "params")?.file ?? null;
    let [train, holdout] = splitSuite(suite, 0.3, salt);
    // Empty train after applying split pins: every case is pinned to holdout, so
    // there is no proposal-visible evidence to optimize against. Stop cleanly.
    if (train.cases.length === 0) {
      return this.abortOutcome(
        emptyScorecard(),
        "improvement stopped: no training cases after applying split pins (every case is " +
          "pinned to `split: holdout`). Pin at least one case to `split: train`, or leave it " +
          "`auto`, so there is evidence to optimize against.",
      );
    }
    // An empty holdout (everything landed in train — pins and/or a small auto pool)
    // is DEGRADED, not distinct: verify against train and warn via the existing
    // not-distinct path below, never run an empty holdout eval.
    const distinctSplit = holdout.cases.length > 0 && !suitesEqual(train, holdout);

    const engine = new EvalEngine(this.runner, this.judge, this.repeats, null, null, "", this.concurrency);
    // Preflight: one cheap run first. A broken entry (import/target error) aborts
    // in seconds with the real error, not after a full baseline eval.
    const preErr = await preflight(this.runner);
    if (preErr) {
      return this.abortOutcome(
        emptyScorecard(),
        abortMessage("the agent failed to run — fix this before improving", preErr),
      );
    }
    let [baselineTrain, baselineRuns] = await engine.runSuiteWithRuns(train);
    // If most agent runs failed to execute, the baseline is measuring a broken
    // environment (throttling, network, bad keys), not agent quality — abort
    // rather than optimize against noise.
    if (agentRunsMostlyFailed(baselineTrain)) {
      const pct = Math.round((100 * baselineTrain.runs_failed) / baselineTrain.runs_total);
      const note = abortMessage(
        `${pct}% of agent runs failed to execute (${baselineTrain.runs_failed}/` +
          `${baselineTrain.runs_total}) — likely an environment or harness problem (API ` +
          `throttling, network, keys, or a broken entry/import), not agent quality`,
        baselineTrain.sample_failure ?? null,
      );
      return this.abortOutcome(baselineTrain, note);
    }
    let baselineHoldout = distinctSplit ? await engine.runSuite(holdout) : baselineTrain;

    // Fold cross-run memory in first: record this run's scores for every case
    // (train AND holdout) and rescore the train baseline by each check's recent
    // mean, so an intermittent failure that passed today's snapshot by luck is
    // still targeted (and fixable).
    const observations = loadObservations(this.observationsPath);
    recordScores(baselineTrain.cases, observations);
    if (distinctSplit) recordScores(baselineHoldout.cases, observations);
    saveObservations(observations, this.observationsPath);
    let recalled = applyHistory(baselineTrain, observations, PASS_THRESHOLD);

    // Train may already be perfect while holdout cases fail — or fail *flakily*,
    // per memory. Rotate the split so those become optimizable; a fresh holdout
    // keeps guarding against overfitting.
    let outcomeNote: string | null = null;
    if (distinctSplit && baselineTrain.cases.every((r) => r.status !== "failed")) {
      const holdoutIds = baselineHoldout.cases.map((r) => r.case_id);
      const flaky = new Set(flakyCaseIds(observations, holdoutIds, PASS_THRESHOLD));
      const holdoutTargets = baselineHoldout.cases
        .filter((r) => r.status === "failed" || flaky.has(r.case_id))
        .map((r) => r.case_id);
      if (holdoutTargets.length) {
        const rotated = findRotationSalt(suite, holdoutTargets, salt);
        if (rotated !== null) {
          salt = rotated;
          [train, holdout] = splitSuite(suite, 0.3, salt);
          [baselineTrain, baselineRuns] = await engine.runSuiteWithRuns(train);
          baselineHoldout = await engine.runSuite(holdout);
          recalled = applyHistory(baselineTrain, observations, PASS_THRESHOLD); // re-blend rotated train
          outcomeNote =
            `rotated the train/holdout split (salt ${salt}) to reach failing or ` +
            `flaky-from-history case(s): ${holdoutTargets.slice(0, 3).join(", ")}`;
        }
      }
    }

    // Establish BASELINE critical evidence before optimizing. A critical case the
    // baseline can't measure (crash/skip) has no constraint to preserve — if left
    // unknown, another case could improve and the candidate be accepted while the
    // critical behavior is silently unguarded (fail-open). Resample; if a critical case
    // STILL can't be measured, ABORT rather than omit it.
    const trainCriticals = new Set(train.cases.filter((c) => c.critical).map((c) => c.id));
    const holdoutCriticals = distinctSplit
      ? new Set(holdout.cases.filter((c) => c.critical).map((c) => c.id))
      : new Set<string>();
    if (trainCriticals.size || holdoutCriticals.size) {
      const est = await establishCriticalBaseline(baselineTrain, trainCriticals, (cid) =>
        this.rerunBaselineCase(train, cid),
      );
      baselineTrain = est.card;
      const unresolved = est.unresolved;
      if (distinctSplit) {
        const estH = await establishCriticalBaseline(baselineHoldout, holdoutCriticals, (cid) =>
          this.rerunBaselineCase(holdout, cid),
        );
        baselineHoldout = estH.card;
        estH.unresolved.forEach((id) => unresolved.add(id));
      }
      if (unresolved.size) {
        const note = abortMessage(
          `critical case(s) have no measurable baseline evidence after resampling ` +
            `(${[...unresolved].sort().join(", ")}) — establish them (fix the check or the ` +
            `harness) before optimizing, so the constraint is never silently omitted`,
          baselineTrain.sample_failure ?? null,
        );
        return this.abortOutcome(baselineTrain, note);
      }
    }

    persistSuite(this.workspace, suite, train, holdout, salt);

    const outcome: ImproveOutcome = {
      baseline: baselineTrain,
      finalScore: baselineTrain.score ?? 0,
      roundsRun: 0,
      accepted: [],
      rejected: [],
      notes: [],
      lessonsAdded: [],
      helpCardsPath: null,
    };
    this.partialOutcome = outcome;
    if (outcomeNote) outcome.notes.push(outcomeNote);
    if (recalled.length) {
      outcome.notes.push(
        `flaky failure(s) recalled from history (passed this run but fail intermittently): ` +
          recalled.slice(0, 3).join(", "),
      );
    }
    if (!distinctSplit) {
      outcome.notes.push(
        "suite too small for a holdout split — scores are train-only; " +
          "add more goals/examples for overfitting protection",
      );
    }

    // Before optimizing toward the judge, check it actually agrees with each
    // quality's own good/bad examples. An unreliable judge means that quality's
    // reward is noise, so warn instead of silently chasing it.
    if (this.judge) {
      const calibration = await calibrateSuite(suite, this.judge);
      for (const bad of unreliableRubrics(calibration)) {
        outcome.notes.push(
          `judge unreliable for quality "${bad.rubric.slice(0, 60)}" — ${bad.detail}; ` +
            "its score is a weak reward signal (run 'agentopt calibrate')",
        );
      }
    }

    const ledger = loadLedger(this.ledgerPath);
    let plateau = 0;
    let lastUnattributed: CaseResult[] = [];

    for (let round = 1; round <= this.maxRounds; round++) {
      outcome.roundsRun = round;
      say(`iteration ${round}/${this.maxRounds}  measuring...`);
      if (round > 1) {
        manifest = this.reloadManifest(manifest);
        this.locator = new PromptLocator(this.projectRoot);
        [baselineTrain, baselineRuns] = await engine.runSuiteWithRuns(train);
        baselineHoldout = distinctSplit ? await engine.runSuite(holdout) : baselineTrain;
      }

      let failures = baselineTrain.cases.filter((r) => r.status === "failed");
      const passed = baselineTrain.cases.filter((r) => r.status === "passed").length;
      const measured = passed + failures.length;
      const score = baselineTrain.score ?? 0;
      say(
        `iteration ${round}/${this.maxRounds}  ` +
          `${score.toFixed(0)}% (${passed}/${measured} checks) -> ${failures.length} failing`,
      );

      if (!failures.length) {
        let grown = false;
        if (this.challenger && !this.challengesAdded) {
          this.challengesAdded = true;
          say("  all checks pass -> generating new challenges...");
          const newCases = await this.challenger(suite);
          if (newCases.length) {
            const beforeCount = train.cases.length;
            train = { ...train, cases: [...train.cases, ...newCases] };
            suite = { ...suite, cases: [...suite.cases, ...newCases] };
            persistSuite(this.workspace, suite, train, holdout, salt);
            say(
              `  suite grew ${beforeCount} -> ${train.cases.length} checks ` +
                "(adversary) - same agent, higher bar",
            );
            outcome.notes.push(
              `generated ${newCases.length} provisional challenge(s) from adversarial ` +
                "probing - veto by removing them from .agentopt/evals/suite.json review",
            );
            [baselineTrain, baselineRuns] = await engine.runSuiteWithRuns(train);
            failures = baselineTrain.cases.filter((r) => r.status === "failed");
            grown = failures.length > 0;
          } else {
            say("  adversary found no new violations");
          }
        }
        if (!grown) {
          outcome.notes.push(`iteration ${round}: all train checks pass`);
          const holdoutFailures = baselineHoldout.cases.filter((r) => r.status === "failed");
          if (distinctSplit && holdoutFailures.length) {
            outcome.notes.push(
              `${holdoutFailures.length} holdout case(s) still fail ` +
                `(${holdoutFailures.slice(0, 3).map((r) => r.case_id).join(", ")}) — ` +
                "the improver never optimizes on holdout; add more examples " +
                "to GOALS.md so the split rotates",
            );
          }
          if (!this.challenger) {
            outcome.notes.push(
              "all checks pass - enable improver.generate_challenges (or run " +
                "with --evolve) to keep improving with generated challenges",
            );
          }
          break;
        }
      }

      const [clusters, unattributed] = clusterFailures(
        failures,
        baselineRuns,
        manifest,
        this.locator,
      );
      lastUnattributed = unattributed;
      for (const orphan of unattributed) {
        outcome.notes.push(
          `${orphan.case_id}: no editable prompt attributable — the workflow tier will handle it`,
        );
      }
      if (!clusters.length) {
        outcome.notes.push(`iteration ${round}: no fixable clusters`);
        break;
      }

      const { accepted: acceptedThisRound, credited: creditedBullets } = await this.runRound(
        clusters.slice(0, this.maxCandidates),
        train,
        holdout,
        baselineTrain,
        baselineHoldout,
        failures.map((f) => f.case_id),
        ledger,
        outcome,
        baselineRuns,
      );

      if (this.playbook.bullets().length) {
        this.playbook.recordOutcome(acceptedThisRound > 0, creditedBullets);
      }

      if (outcome.accepted.length >= this.maxChanges) {
        outcome.notes.push(
          `stopping: version cap reached (${this.maxChanges} created this run) — ` +
            "review them with 'agentopt show', then run improve again",
        );
        break;
      }

      plateau = acceptedThisRound ? 0 : plateau + 1;
      if (plateau >= this.plateauStop) {
        outcome.notes.push(`stopping: stuck for ${plateau} iteration(s) with no verified fix`);
        break;
      }
    }

    if (lastUnattributed.length && this.tier4) {
      const result = await this.tier4.attempt(train, lastUnattributed);
      outcome.notes.push(`tier-4: ${result.reason}`);
      if (result.accepted) {
        outcome.finalScore = Math.max(outcome.finalScore, baselineTrain.score ?? outcome.finalScore);
        ledger.attempts.push({
          id: `att_${randomBytes(4).toString("hex")}`,
          ts: new Date().toISOString(),
          tier: 4,
          target_id: "workflow",
          description: "delegated code fix (tier 4)",
          outcome: "applied",
          version: result.version,
          decision_ref: result.decisionRef ?? null,
        });
        if (result.version) {
          this.writeReport(result.version, {
            title: "delegated code fix (tier 4)",
            tier: 4,
            targetId: "workflow",
            targetFile: null,
            reason: result.reason,
            files: result.changedFiles,
          });
        }
      }
    } else if (lastUnattributed.length) {
      outcome.notes.push(
        `${lastUnattributed.length} failure(s) had no editable prompt to fix — ` +
          "see .agentopt/review/ and GOALS.md examples for ways to unblock them",
      );
    }

    saveLedger(ledger, this.ledgerPath);
    this.emitHelpCards(outcome);
    return outcome;
  }

  private async runRound(
    clusters: FailureCluster[],
    train: EvalSuite,
    holdout: EvalSuite,
    baselineTrain: Scorecard,
    baselineHoldout: Scorecard,
    failingIds: string[],
    ledger: Ledger,
    outcome: ImproveOutcome,
    baselineRuns: Map<string, RunData[]>,
  ): Promise<{ accepted: number; credited: Set<string> }> {
    let acceptedCount = 0;
    const credited = new Set<string>();

    for (const cluster of clusters) {
      const entry = cluster.entry;
      const current = this.currentContent(entry);
      if (!current) {
        outcome.notes.push(`${entry.id}: source moved — run 'agentopt observe'`);
        continue;
      }

      const attemptsForEntry = attemptsForTarget(ledger, entry.id);
      const past = attemptsForEntry.filter((a) => a.outcome === "discarded");
      const applied = attemptsForEntry.filter((a) => a.outcome === "applied");
      let guidance: string | null = null;
      if (past.length && this.searcher) {
        // Cache per failing-goal set so a cluster stuck across rounds searches once.
        const cacheKey = cluster.failures.map((f) => f.source).join("|");
        if (this.searchCache.has(cacheKey)) {
          guidance = this.searchCache.get(cacheKey) ?? null;
        } else {
          guidance = await this.searcher(buildSearchProblem(cluster.failures, past));
          this.searchCache.set(cacheKey, guidance);
        }
        if (guidance) outcome.notes.push(`${entry.id}: consulted web guidance (stuck)`);
      }
      const diagnostics = diagnoseRuntime(cluster.failures, baselineRuns);
      // EXPLORE: a diverse population of candidates, not one greedy idea.
      let candidates = await this.reflector.proposeCandidates(
        this.explore,
        cluster.failures,
        entry,
        current,
        past,
        guidance,
        applied,
        diagnostics,
      );
      // Exact dedup: drop any candidate structurally identical (target + kind +
      // normalized content) to one already tried and rejected/rolled-back — the LLM
      // sometimes re-proposes a reworded duplicate the "already tried" text list
      // didn't catch, and re-verifying it just burns budget.
      const triedFingerprints = new Set(
        attemptsForEntry
          .filter((a) => a.fingerprint && (a.outcome === "discarded" || a.outcome === "rolled_back"))
          .map((a) => a.fingerprint),
      );
      if (triedFingerprints.size) {
        const fresh = candidates.filter((c) => !triedFingerprints.has(candidateFingerprint(c)));
        const skipped = candidates.length - fresh.length;
        if (skipped) outcome.notes.push(`${entry.id}: skipped ${skipped} already-tried candidate(s)`);
        candidates = fresh;
      }
      if (!candidates.length) {
        outcome.notes.push(`${entry.id}: no valid proposal from the optimizer`);
        this.markStuck(cluster, "no valid proposal");
        continue;
      }

      // EXPERIMENT: cheaply probe each candidate on the failing cases (one run)
      // and observe which actually improves — the agent's own experiment,
      // rather than trusting the first idea.
      const probed: Array<{
        candidate: Candidate;
        overlay: Overlay;
        probe: number;
        tokens: number | null;
        latency: number | null;
      }> = [];
      for (const candidate of candidates) {
        const overlay = overlayFor(candidate, entry, current);
        const { improvement, tokens, latency } = await this.probeCandidate(overlay, train, failingIds, baselineTrain);
        probed.push({ candidate, overlay, probe: improvement, tokens, latency });
      }
      // Rank by improvement; among equally-promising candidates, prefer fewer tokens,
      // THEN lower latency (lexicographic — the two signals stay separate, never a
      // unit-mixing scalar). Unavailable cost sorts last (worst tiebreak). Improvement
      // is quantized (9 dp) so the sort AND the audit basis below agree on when two
      // candidates are "equally promising".
      const INF = Number.POSITIVE_INFINITY;
      const q = (x: number) => Math.round(x * 1e9) / 1e9;
      const tok = (p: (typeof probed)[number]) => p.tokens ?? INF;
      const lat = (p: (typeof probed)[number]) => p.latency ?? INF;
      probed.sort((a, b) => q(b.probe) - q(a.probe) || tok(a) - tok(b) || lat(a) - lat(b));
      if (candidates.length > 1) {
        // Report the ACTUAL decisive comparison between the winner and the runner-up,
        // using the SAME normalized values the sort used (missing cost == +inf):
        // improvement if it broke them apart, else the cost signal (tokens, then
        // latency) that differed — not just "the winner has tokens".
        const top = probed[0]!;
        const second = probed[1]!;
        const basis =
          q(top.probe) > q(second.probe)
            ? "improvement"
            : tok(top) !== tok(second)
              ? "tokens"
              : lat(top) !== lat(second)
                ? "latency"
                : "tie (stable order)";
        say(
          `  exploring ${probed.length} candidate(s) for ${entry.id} — ` +
            `selected ${top.candidate.id} (probe ${top.probe >= 0 ? "+" : ""}${(top.probe * 100).toFixed(0)}%, ` +
            `decided by: ${basis})`,
        );
      }

      // VERIFY the promising candidates best-first; apply the first that
      // survives the full gate. Candidates the experiment showed no gain are
      // skipped (unless none helped, then the best is still verified).
      const helped = probed.filter((p) => p.probe > 0.01);
      const toVerify = helped.length ? helped : probed.slice(0, 1);
      let acceptedThis = false;
      for (const { candidate, overlay } of toVerify) {
        const result = await this.gate.verify(
          candidate,
          overlay,
          train,
          holdout,
          baselineTrain,
          baselineHoldout,
          failingIds,
        );
        if (!result.accepted) {
          // Persist the rejected candidate's structured decision too (not only
          // accepted ones), and link it from the ledger — a rejection is a real gate
          // decision worth auditing.
          const decisionRef = this.persistRejectedDecision(result);
          ledger.attempts.push(attemptFrom(result, null, decisionRef));
          outcome.rejected.push(result);
          continue;
        }
        let change = this.applyIfPossible({ candidate, overlay, result, entry, version: null }, outcome);
        if (change.version === null && candidate.kind === "params" && candidate.param_overrides) {
          change = await this.applyParamsChange(change, train, outcome);
        }
        ledger.attempts.push(attemptFrom(result, change.version));
        if (change.version !== null || candidate.kind === "params") {
          const label = change.version ?? "recommendation";
          say(
            `  fixing ${entry.id}... verified: ` +
              `${result.baseline_score.toFixed(0)}% -> ${(result.candidate_score ?? 0).toFixed(0)}% ` +
              `(${label} applied)`,
          );
          acceptedCount++;
          outcome.accepted.push(change);
          outcome.finalScore = Math.max(outcome.finalScore, result.candidate_score ?? 0);
          this.stuckRounds.delete(entry.id);
          await this.distill(change, cluster, outcome);
        }
        acceptedThis = true;
        break;
      }
      // ESCALATE: no prompt/param edit survived for this cluster. A structural
      // code edit now competes on the same significance standard — chosen when
      // it is genuinely the best move, not only for orphan failures.
      if (!acceptedThis && this.tier4?.available()) {
        acceptedThis = await this.tryCodeCandidate(cluster, train, holdout, ledger, outcome);
        if (acceptedThis) acceptedCount++;
      }
      if (acceptedThis) {
        // Scope reward credit to the lessons that plausibly bore on this fix
        // (content-word overlap with the fixed cluster's failures) — computed
        // after distillation so a freshly-learned lesson counts.
        const clusterText = cluster.failures.map((f) => `${f.source} ${f.detail}`).join(" ");
        for (const id of relevantBullets(this.playbook.bullets(), clusterText)) credited.add(id);
      } else {
        say(
          `  trying ${entry.id}... discarded ` +
            `(${probed.length} candidate(s) explored, none survived verification)`,
        );
        this.markStuck(cluster, `${probed[0]!.candidate.description} — none of ${probed.length} survived`);
      }
    }

    return { accepted: acceptedCount, credited };
  }

  /**
   * Try a delegated code edit for a cluster no prompt/param change could fix.
   * The delegate applies the edit to real source, and it survives only on the
   * same significance-graded gate (real gain, no significant regression, no
   * holdout overfitting) — otherwise every touched file is reverted.
   */
  private async tryCodeCandidate(
    cluster: FailureCluster,
    train: EvalSuite,
    holdout: EvalSuite,
    ledger: Ledger,
    outcome: ImproveOutcome,
  ): Promise<boolean> {
    const entry = cluster.entry;
    say(`  ${entry.id}: no prompt/param fix survived — trying a code edit (tier 4)...`);
    const distinct = !suitesEqual(holdout, train);
    const result = await this.tier4!.attemptForCluster(
      train,
      distinct ? holdout : null,
      cluster.failures,
      entry.file ?? null,
    );
    outcome.notes.push(`${entry.id} tier-4: ${result.reason}`);
    if (!result.accepted) return false;

    const candidate: Candidate = {
      id: `code_${randomBytes(4).toString("hex")}`,
      tier: 4,
      entry_id: entry.id,
      kind: "code",
      description: `delegated code fix (${result.changedFiles.join(", ") || "workflow"})`,
      reasoning: "",
      new_content: null,
      param_overrides: null,
      code_instructions: result.reason,
    };
    const gateResult: GateResult = {
      candidate,
      accepted: true,
      reason: result.reason,
      baseline_score: result.baselineScore ?? 0,
      candidate_score: result.candidateScore,
      holdout_before: null,
      holdout_after: null,
      full_before: null,
      full_after: null,
    };
    const change: VerifiedChange = {
      candidate,
      overlay: { rules: [] },
      result: gateResult,
      entry,
      version: result.version,
      changed_files: result.changedFiles,
    };
    ledger.attempts.push({
      id: `att_${randomBytes(4).toString("hex")}`,
      ts: new Date().toISOString(),
      tier: 4,
      target_id: entry.id,
      description: "delegated code fix (tier 4)",
      outcome: "applied",
      version: result.version,
      decision_ref: result.decisionRef ?? null,
    });
    if (result.version) this.writeVersionReport(result.version, change, result.decisionRef);
    say(
      `  fixing ${entry.id}... code edit verified: ` +
        `${(result.baselineScore ?? 0).toFixed(0)}% -> ${(result.candidateScore ?? 0).toFixed(0)}% ` +
        `(${result.version ?? "applied"})`,
    );
    outcome.accepted.push(change);
    outcome.finalScore = Math.max(outcome.finalScore, result.candidateScore ?? 0);
    this.stuckRounds.delete(entry.id);
    await this.distill(change, cluster, outcome);
    return true;
  }

  /**
   * A cheap experiment: run the failing cases once under the candidate's
   * overlay and return the mean score improvement over baseline. Positive means
   * the candidate actually helped — used to rank the population before the
   * expensive full-gate verification.
   */
  private async probeCandidate(
    overlay: Overlay,
    train: EvalSuite,
    failingIds: string[],
    baselineTrain: Scorecard,
  ): Promise<{ improvement: number; tokens: number | null; latency: number | null }> {
    return this.probeCandidateImpl(overlay, train, failingIds, baselineTrain);
  }

  /** Re-run the BASELINE (no overlay) on a single case — for establishing a critical
   * case's baseline evidence. Null when the case isn't in the suite. */
  private async rerunBaselineCase(suite: EvalSuite, caseId: string): Promise<CaseResult | null> {
    const evalCase = suite.cases.find((c) => c.id === caseId);
    if (!evalCase) return null;
    const [gradeSuite, runPlan] = singleCasePlan(suite, evalCase);
    const cards = await new EvalEngine(
      this.runner,
      this.judge,
      RESAMPLE_STEP,
      null,
      null,
      "",
      this.concurrency,
    ).runSuite(gradeSuite, runPlan);
    return cards.cases[0] ?? null;
  }

  private async probeCandidateImpl(
    overlay: Overlay,
    train: EvalSuite,
    failingIds: string[],
    baselineTrain: Scorecard,
  ): Promise<{ improvement: number; tokens: number | null; latency: number | null }> {
    const cases = train.cases.filter((c) => failingIds.includes(c.id));
    if (!cases.length) return { improvement: 0, tokens: null, latency: null };
    // Grade only the failing cases, but EXECUTE the full input set on their surfaces (as
    // the baseline did) so a null-input rule/quality is ranked like-for-like against its
    // baseline evidence, not a reduced/synthetic run set (#3).
    const surfaces = new Set(cases.map((c) => c.surface));
    const runPlan: EvalSuite = { ...train, cases: train.cases.filter((c) => surfaces.has(c.surface)) };
    const probe = await new EvalEngine(this.runner, this.judge, 1, overlay, null, "", this.concurrency).runSuite(
      { ...train, cases },
      runPlan,
    );
    const baseByCase = new Map(baselineTrain.cases.map((c) => [c.case_id, c.score]));
    let deltaSum = 0;
    let counted = 0;
    for (const result of probe.cases) {
      if (result.status === "skipped") continue;
      deltaSum += result.score - (baseByCase.get(result.case_id) ?? 0);
      counted += 1;
    }
    const improvement = counted ? deltaSum / counted : 0;
    // Return tokens and latency SEPARATELY — never collapsed into one scalar (which
    // would compare 1000 tokens against 50 ms). The ranker breaks ties lexicographically.
    return { improvement, tokens: probe.total_tokens ?? null, latency: probe.total_latency_ms ?? null };
  }

  private async distill(
    change: VerifiedChange,
    cluster: FailureCluster,
    outcome: ImproveOutcome,
  ): Promise<void> {
    // Every accepted change is a lesson worth keeping — including params and
    // code. The most reusable insight this optimizer ever learned ("reasoning
    // models need a generous max_tokens or they truncate to empty") is a params
    // change; a code fix's lesson lives in its diff, not its file name.
    const detail = this.changeDetail(change);
    const lesson = await this.reflector.distillLesson(change.candidate, cluster.failures, detail);
    if (lesson && this.playbook.add(lesson) !== null) {
      outcome.lessonsAdded.push(lesson);
    }
  }

  /** The concrete substance of a change, for grounding its distilled lesson. */
  private changeDetail(change: VerifiedChange): string {
    const candidate = change.candidate;
    if (candidate.kind === "prompt") return `NEW PROMPT:\n${candidate.new_content ?? ""}`;
    if (candidate.kind === "params") {
      return `PARAM CHANGE: ${JSON.stringify(candidate.param_overrides)}`;
    }
    // Code: the lesson is in what actually changed — feed a compact diff.
    const diff = this.compactDiff(change);
    return diff ? `CODE DIFF:\n${diff}` : `CODE CHANGE: ${candidate.description}`;
  }

  /** A size-capped unified diff of the files a code change rewrote. */
  private compactDiff(change: VerifiedChange): string | null {
    if (!change.version || !change.changed_files?.length) return null;
    const lines: string[] = [];
    for (const file of change.changed_files) {
      const preImage = join(this.workspace.historyDir, change.version, "files", file);
      const live = join(this.projectRoot, file);
      if (!existsSync(preImage) || !existsSync(live)) continue;
      lines.push(...unifiedDiff(readFileSync(preImage, "utf-8"), readFileSync(live, "utf-8"), file, file));
      if (lines.length > 120) break; // keep the distill prompt small
    }
    return lines.length ? lines.slice(0, 120).join("\n") : null;
  }

  private markStuck(cluster: FailureCluster, attemptDescription: string): void {
    const entryId = cluster.entry.id;
    this.stuckRounds.set(entryId, (this.stuckRounds.get(entryId) ?? 0) + 1);
    const attempts = this.clusterAttempts.get(entryId) ?? [];
    attempts.push(attemptDescription);
    this.clusterAttempts.set(entryId, attempts);
  }

  private emitHelpCards(outcome: ImproveOutcome): void {
    const cards: HelpCard[] = [...this.stuckRounds.entries()]
      .filter(([, rounds]) => rounds >= STUCK_THRESHOLD)
      .map(([entryId]) => ({
        entryId,
        failingGoals: [],
        attemptsTried: this.clusterAttempts.get(entryId) ?? [],
      }));
    if (cards.length) {
      outcome.helpCardsPath = writeHelpCards(this.workspace, cards);
    } else {
      // Nothing is stuck — a lingering card from an earlier run would report
      // an already-solved problem.
      const stale = join(this.workspace.reviewDir, "help_needed.md");
      if (existsSync(stale)) unlinkSync(stale);
    }
  }

  /** Write a verified prompt change to its file; params stay advisory. */
  private applyIfPossible(change: VerifiedChange, outcome: ImproveOutcome): VerifiedChange {
    // Params changes have no literal to rewrite — the caller applies them to
    // source via the tier-4 delegate (or falls back to a recommendation).
    if (!this.writer || change.candidate.kind !== "prompt") {
      return change;
    }
    try {
      const result = change.result;
      const version = this.writer.apply(change.candidate, change.entry, {
        scoreBefore: result.full_before ?? result.baseline_score,
        scoreAfter: result.full_after ?? result.candidate_score,
      });
      this.writeVersionReport(version, change);
      return { ...change, version };
    } catch (error) {
      if (!(error instanceof WriterError)) throw error;
      outcome.notes.push(`${change.entry.id}: verified but not applied — ${error.message}`);
      return change;
    }
  }

  /**
   * Apply a verified params change (e.g. temperature) to source via the tier-4
   * delegate — the structural edit the literal writer can't make. Falls back to
   * a recommendation when there is no delegate or the edit can't be verified.
   */
  private async applyParamsChange(
    change: VerifiedChange,
    suite: EvalSuite,
    outcome: ImproveOutcome,
  ): Promise<VerifiedChange> {
    const overrides = change.candidate.param_overrides ?? {};
    const recommend = (why: string) =>
      outcome.notes.push(
        `${change.entry.id}: verified params change ${JSON.stringify(overrides)} — ${why}; ` +
          "set it on your model call",
      );
    const current = this.currentContent(change.entry);

    // First try the deterministic AST writer: no delegate, no git, works with
    // any backend. The overlay already verified the change, so a faithful AST
    // edit of the same call needs no re-run. Falls through on a miss.
    if (this.writer) {
      try {
        const version = this.writer.applyParams(
          overrides,
          change.entry,
          current ? current.text : null,
          change.entry.line ?? null,
          {
            scoreBefore: change.result.baseline_score,
            scoreAfter: change.result.candidate_score,
          },
          change.candidate.description,
        );
        outcome.notes.push(
          `${change.entry.id}: params ${JSON.stringify(overrides)} applied to code as ${version} (AST)`,
        );
        const applied = { ...change, version };
        this.writeVersionReport(version, applied);
        return applied;
      } catch {
        // could not locate the call by AST — try the code delegate below
      }
    }

    if (!this.tier4) {
      recommend("no code delegate available to apply it");
      return change;
    }
    // Point the delegate at THIS candidate's own call — the one sending its
    // prompt — not "the main call" (which biased edits to the tool-using coach).
    const target = {
      file: change.entry.file ?? this.paramsHintFile,
      line: change.entry.line ?? null,
      promptSnippet: current ? current.text.slice(0, 240) : null,
    };
    const applied = await this.tier4.applyParams(suite, overrides, target);
    if (applied.version) {
      outcome.notes.push(
        `${change.entry.id}: params ${JSON.stringify(overrides)} applied to code as ${applied.version}`,
      );
      const change4 = { ...change, version: applied.version };
      this.writeVersionReport(applied.version, change4);
      return change4;
    }
    recommend(`not auto-applied (${applied.reason})`);
    return change;
  }

  /** One human-readable report per version: what changed, why, evidence. */
  /** Persist a REJECTED overlay candidate's structured decision + evidence, named by a
   * fresh rejected-* id (no version — it was not applied). Returns the ref, or null
   * only if the gate attached no inputs (every rejection now does). */
  private persistRejectedDecision(result: GateResult): string | null {
    const di = result.decisionInputs;
    if (di == null) return null;
    return persistDecision(this.workspace.reportsDir, `rejected-${randomBytes(6).toString("hex")}`, {
      ...di,
      createdAt: new Date().toISOString(),
    }).decisionRef;
  }

  private writeVersionReport(
    version: string,
    change: VerifiedChange,
    decisionRef?: string | null,
  ): void {
    const entry = change.entry;
    const result = change.result;
    // The overlay gate attaches its decision INPUTS; persist the structured decision +
    // its evidence scorecards named by this version, and link it (the code path already
    // persisted its own and passes decisionRef explicitly).
    if (decisionRef == null && result.decisionInputs != null) {
      decisionRef = persistDecision(this.workspace.reportsDir, version, {
        ...result.decisionInputs,
        createdAt: new Date().toISOString(),
      }).decisionRef;
    }
    // The full-suite score at accept time (what meta.json / the versions list
    // show) — re-sampled each run, so surface it labelled as sampled alongside
    // the gate verdict. Falls back to the train score when no full pass was run.
    this.writeReport(version, {
      title: change.candidate.description,
      tier: change.candidate.tier,
      targetId: entry.id,
      targetFile: entry.file ?? null,
      reason: result.reason,
      sampledBefore: result.full_before ?? result.baseline_score,
      sampledAfter: result.full_after ?? result.candidate_score,
      reasoning: change.candidate.reasoning || null,
      // A prompt/param edit touches one located file; a delegated code edit can
      // touch several — report every file the change actually rewrote.
      files: change.changed_files ?? (entry.file ? [entry.file] : []),
      decisionRef: decisionRef ?? null,
    });
  }

  /** Render a version's markdown report, diffing every file it rewrote. */
  private writeReport(
    version: string,
    report: {
      title: string;
      tier: number;
      targetId: string;
      targetFile: string | null;
      reason: string;
      sampledBefore?: number | null;
      sampledAfter?: number | null;
      reasoning?: string | null;
      files: string[];
      decisionRef?: string | null;
    },
  ): void {
    // Resolve each file's pre-image (snapshotted under history/<version>/files)
    // and live source; the pure renderer turns them into the report text.
    const files: ReportFile[] = report.files.map((path) => {
      const preImage = join(this.workspace.historyDir, version, "files", path);
      const live = join(this.projectRoot, path);
      return {
        path,
        before: existsSync(preImage) ? readFileSync(preImage, "utf-8") : null,
        after: existsSync(live) ? readFileSync(live, "utf-8") : null,
      };
    });
    const appliedAt = `${new Date().toISOString().slice(0, 19)}Z`;
    const text = renderVersionReport({
      version,
      title: report.title,
      targetId: report.targetId,
      targetFile: report.targetFile,
      evidence: report.reason,
      sampledBefore: report.sampledBefore ?? null,
      sampledAfter: report.sampledAfter ?? null,
      why: report.reasoning ?? null,
      appliedAt,
      files,
    });
    const targetDir = join(this.workspace.reportsDir, "versions");
    mkdirSync(targetDir, { recursive: true });
    writeFileSync(join(targetDir, `${version}.md`), text, "utf-8");
    // The machine-readable source of truth alongside the human markdown.
    const structured = buildVersionReport({
      version,
      title: report.title,
      tier: report.tier,
      targetId: report.targetId,
      evidence: report.reason,
      appliedAt,
      files,
      rollbackCommand: `agentopt rollback ${version}`,
      decisionRef: report.decisionRef ?? null,
    });
    writeFileSync(join(targetDir, `${version}.json`), JSON.stringify(structured, null, 2), "utf-8");
    // A self-contained, shareable HTML view rendered from the same JSON.
    writeFileSync(join(targetDir, `${version}.html`), renderVersionHtml(structured), "utf-8");
  }

  private currentContent(entry: ManifestEntry): EntryContent | null {
    if (!entry.file || !entry.node_path) return null;
    const located = this.locator.candidateAt(entry.file, entry.node_path);
    if (!located) return null;
    return { text: located.template || located.staticValue || "", pattern: located.pattern };
  }

  private reloadManifest(fallback: Manifest): Manifest {
    const manifestPath = this.workspace.manifestFile;
    if (existsSync(manifestPath)) {
      return ManifestSchema.parse(JSON.parse(readFileSync(manifestPath, "utf-8")));
    }
    return fallback;
  }
}

/** The enriched query for web guidance: what's failing, how, and what to avoid. */
/**
 * Runtime tells about the failing runs that the prompt text alone can't reveal.
 * The prime one: an EMPTY model completion — the reflector would otherwise
 * blame the wording, when the real cause is a too-small max_tokens starving the
 * response (acute for reasoning models, which spend tokens before answering).
 */
export function diagnoseRuntime(
  failures: CaseResult[],
  runsByInput: Map<string, RunData[]>,
): string | null {
  let sawEmpty = false;
  let minMaxTokens: number | null = null;
  for (const failure of failures) {
    for (const call of llmCallsForFailure(failure, runsByInput)) {
      const content = (call.response?.content ?? "").trim();
      if (content !== "" || (call.response?.tool_calls?.length ?? 0) > 0) continue;
      sawEmpty = true;
      const cap = call.params?.max_output_tokens;
      if (typeof cap === "number") minMaxTokens = minMaxTokens === null ? cap : Math.min(minMaxTokens, cap);
    }
  }
  if (!sawEmpty) return null;
  return (
    "the model returned EMPTY output on at least one failing case" +
    (minMaxTokens !== null ? ` while max_tokens was ${minMaxTokens}` : "") +
    " — the call was likely truncated before it could answer (reasoning models " +
    "spend tokens thinking first). If the code then falls back to a default on empty " +
    "output, that explains the wrong result. Raise max_tokens GENEROUSLY (e.g. 2048–4096, " +
    "not a small bump) — the hidden reasoning can be far longer than the current limit, " +
    "so a timid increase may still truncate to empty."
  );
}

function buildSearchProblem(failures: CaseResult[], tried: Attempt[]): string {
  const list = (lines: string[]) => (lines.length ? lines.join("\n") : "- (none)");
  const goals = list(failures.slice(0, 3).map((f) => `- ${f.source}`));
  const wrong = list(failures.slice(0, 3).map((f) => `- ${f.detail}`));
  const already = list(tried.slice(-5).map((a) => `- ${a.description}`));
  return (
    `GOAL(S) FAILING:\n${goals}\n\n` +
    `WHAT THE AGENT DOES WRONG:\n${wrong}\n\n` +
    `ALREADY TRIED (do NOT re-suggest):\n${already}`
  );
}

function attemptFrom(
  result: GateResult,
  version: string | null,
  decisionRef: string | null = null,
): Attempt {
  // An accepted overlay change persists its decision as decisions/<version>.json (see
  // persistDecision); a rejected one gets a decisions/rejected-*.json path passed in.
  const ref = decisionRef ?? (result.accepted && version ? `decisions/${version}.json` : null);
  return {
    id: `att_${randomBytes(4).toString("hex")}`,
    ts: new Date().toISOString(),
    tier: result.candidate.tier,
    target_id: result.candidate.entry_id,
    description: result.candidate.description,
    outcome: result.accepted ? "applied" : "discarded",
    score_delta:
      result.candidate_score !== null ? result.candidate_score - result.baseline_score : null,
    version,
    reason: result.accepted ? null : result.reason,
    fingerprint: candidateFingerprint(result.candidate),
    decision_ref: ref,
  };
}

function emptyScorecard(): Scorecard {
  return {
    created_at: new Date().toISOString(),
    score: null,
    measured_cases: 0,
    skipped_cases: 0,
    cases: [],
    warnings: [],
    runs_total: 0,
    runs_failed: 0,
  };
}

/** A minimal unified diff — line-level, enough for the version report. */
