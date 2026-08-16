/** N6: gateway retries, HTTP backends, the judge, and the reflector. */

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  BackendUnavailableError,
  ClaudeCodeBackend,
  CodexBackend,
  createJudgeBackend,
  GatewayError,
  HttpBackend,
  LlmJudge,
  RateLimitedError,
  searchGuidance,
  TransientError,
  withRetries,
} from "../src/gateway.js";
import { CachingJudge } from "../src/evals/commandJudge.js";
import type { Judge } from "../src/evals/engine.js";
import { Reflector, stripLessonPreamble } from "../src/improver/reflector.js";
import type { ManifestEntry } from "../src/models/manifest.js";
import { ScriptedBackend } from "./helpers.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

const noSleep = async () => {};

describe("withRetries", () => {
  it("retries transient errors and returns the eventual success", async () => {
    let calls = 0;
    const result = await withRetries(
      async () => {
        calls++;
        if (calls < 3) throw new TransientError("blip");
        return "ok";
      },
      4,
      noSleep,
    );
    expect(result).toBe("ok");
    expect(calls).toBe(3);
  });

  it("gives up after the attempt budget and keeps the error class", async () => {
    await expect(
      withRetries(async () => Promise.reject(new RateLimitedError("429")), 2, noSleep),
    ).rejects.toThrow(RateLimitedError);
  });

  it("never retries non-transient errors", async () => {
    let calls = 0;
    await expect(
      withRetries(
        async () => {
          calls++;
          throw new GatewayError("bad request");
        },
        4,
        noSleep,
      ),
    ).rejects.toThrow(GatewayError);
    expect(calls).toBe(1);
  });
});

describe("HttpBackend", () => {
  it("speaks openai-compatible chat completions", async () => {
    let sent: Record<string, unknown> = {};
    let headers: Record<string, string> = {};
    globalThis.fetch = vi.fn(async (url: unknown, init?: RequestInit) => {
      sent = JSON.parse(String(init?.body)) as Record<string, unknown>;
      headers = Object.fromEntries(Object.entries(init?.headers ?? {}));
      expect(String(url)).toBe("https://llm.example/v1/chat/completions");
      return new Response(
        JSON.stringify({ choices: [{ message: { content: "reply text" } }] }),
        { headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;

    const backend = new HttpBackend("cloud", "https://llm.example/v1", "sk-test", "my-model");
    const reply = await backend.complete([
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
    ]);
    expect(reply).toBe("reply text");
    expect(sent.model).toBe("my-model");
    expect(headers.authorization).toBe("Bearer sk-test");
  });

  it("speaks anthropic format with system extracted", async () => {
    let sent: Record<string, unknown> = {};
    globalThis.fetch = vi.fn(async (url: unknown, init?: RequestInit) => {
      sent = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(String(url)).toBe("https://api.anthropic.com/v1/messages");
      return new Response(
        JSON.stringify({ content: [{ type: "text", text: "claude says hi" }] }),
        { headers: { "content-type": "application/json" } },
      );
    }) as unknown as typeof fetch;

    const backend = new HttpBackend("anthropic", "https://api.anthropic.com", "key", "claude-sonnet-5", true);
    const reply = await backend.complete([
      { role: "system", content: "be brief" },
      { role: "user", content: "hi" },
    ]);
    expect(reply).toBe("claude says hi");
    expect(sent.system).toBe("be brief");
    expect((sent.messages as unknown[]).length).toBe(1);
  });

  it("retries a 429 then succeeds", async () => {
    let calls = 0;
    globalThis.fetch = vi.fn(async () => {
      calls++;
      if (calls === 1) return new Response("slow down", { status: 429 });
      return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }));
    }) as unknown as typeof fetch;

    // withRetries sleeps for real; patch setTimeout to fire immediately
    vi.spyOn(globalThis, "setTimeout").mockImplementation(((fn: () => void) => {
      fn();
      return 0 as unknown as NodeJS.Timeout;
    }) as typeof setTimeout);

    const backend = new HttpBackend("cloud", "https://llm.example/v1", "k", "m");
    expect(await backend.complete([{ role: "user", content: "x" }])).toBe("ok");
    expect(calls).toBe(2);
  });

  it("requireModel resolves overrides/defaults and fails loudly when model-less", () => {
    const withDefault = new HttpBackend("cloud", "u", "k", "default-model");
    expect(withDefault.requireModel(null)).toBe("default-model");
    expect(withDefault.requireModel("override")).toBe("override");

    const modelless = new HttpBackend("ollama", "u", "k", null);
    expect(() => modelless.requireModel(null)).toThrow(BackendUnavailableError);
    expect(() => modelless.requireModel(null)).toThrow(/improver\.optimizer/);
    expect(modelless.requireModel("llama3.1")).toBe("llama3.1");

    // Claude Code always has a CLI default → "" means "no --model flag", never throws.
    expect(new ClaudeCodeBackend().requireModel(null)).toBe("");
    expect(new ClaudeCodeBackend().requireModel("claude-opus-4-8")).toBe("claude-opus-4-8");

    // Codex likewise has a CLI default → "" means "no --model flag".
    expect(new CodexBackend().requireModel(null)).toBe("");
    expect(new CodexBackend().requireModel("gpt-5.5-codex")).toBe("gpt-5.5-codex");
  });
});

describe("LlmJudge", () => {
  it("builds a rubric prompt and parses a graded 0-100 score into [0,1]", async () => {
    const backend = new ScriptedBackend(['{"score": 40, "reasoning": "too curt"}']);
    const judge = new LlmJudge(backend);
    const verdict = await judge.evaluate({
      rubric: "Responses are polite.",
      output: "no.",
      inputText: "help me",
      good: ["Happy to help!"],
      bad: ["not my problem"],
      glossary: { fan: "a paying subscriber" },
    });
    expect(verdict).toEqual({ score: 0.4, reasoning: "too curt" });
    const sent = backend.prompts[0]!.map((m) => m.content).join("\n");
    expect(sent).toContain("RUBRIC:\nResponses are polite.");
    expect(sent).toContain('"fan" means a paying subscriber');
    expect(sent).toContain("Happy to help!");
    expect(sent).toContain("AGENT OUTPUT TO GRADE:\nno.");
  });

  it("accepts a 0-1 score directly and tolerates a legacy boolean verdict", async () => {
    const graded = new LlmJudge(new ScriptedBackend(['{"score": 0.85, "reasoning": "good"}']));
    expect((await graded.evaluate(EMPTY_JUDGE_INPUT)).score).toBe(0.85);
    const legacy = new LlmJudge(new ScriptedBackend(['{"passed": true, "reasoning": "ok"}']));
    expect((await legacy.evaluate(EMPTY_JUDGE_INPUT)).score).toBe(1);
  });

  it("rejects replies with neither a score nor a verdict", async () => {
    const judge = new LlmJudge(new ScriptedBackend(['{"reasoning": "hmm"}']));
    await expect(judge.evaluate(EMPTY_JUDGE_INPUT)).rejects.toThrow(GatewayError);
  });

  it("#4: a confident verdict costs a single call even with samples enabled", async () => {
    // 0.9 is far above the 0.6 threshold — no ambiguity, so no re-sampling.
    const backend = new ScriptedBackend(['{"score": 90, "reasoning": "clearly good"}']);
    const judge = new LlmJudge(backend, null, 3);
    const verdict = await judge.evaluate(EMPTY_JUDGE_INPUT);
    expect(verdict.score).toBe(0.9);
    expect(backend.prompts).toHaveLength(1); // did NOT spend extra samples
  });

  it("#4: a borderline verdict re-samples and returns the median (self-consistency)", async () => {
    // 0.6 sits on the threshold — ambiguous, so it re-samples up to 3 and takes
    // the median (0.5, 0.6, 0.7 -> 0.6), immune to a single outlier flip.
    const backend = new ScriptedBackend([
      '{"score": 60, "reasoning": "borderline"}',
      '{"score": 50, "reasoning": "a bit low"}',
      '{"score": 70, "reasoning": "a bit high"}',
    ]);
    const judge = new LlmJudge(backend, null, 3);
    const verdict = await judge.evaluate(EMPTY_JUDGE_INPUT);
    expect(verdict.score).toBeCloseTo(0.6, 5);
    expect(backend.prompts).toHaveLength(3); // spent the extra samples
    // First sample is deterministic (temp 0); the extra samples vary (temp > 0).
    expect(verdict.reasoning).toContain("median of 3 samples");
  });
});

const EMPTY_JUDGE_INPUT = {
  rubric: "r",
  output: "o",
  inputText: null,
  good: [] as string[],
  bad: [] as string[],
  glossary: {},
};

describe("createJudgeBackend (provider-generic judge backend)", () => {
  it("points a cloud judge at a given endpoint using a NAMED key env var", async () => {
    process.env.AGENTOPT_TEST_FW_KEY = "fw_test_123";
    try {
      const choice = await createJudgeBackend({
        backend: "claude-code",
        judge_backend: "cloud",
        judge_base_url: "https://api.fireworks.ai/inference/v1",
        judge_api_key_env: "AGENTOPT_TEST_FW_KEY",
      });
      // The key came from FIREWORKS_API_KEY-style env, not OPENAI_API_KEY; the
      // endpoint is the provider we named, not OpenAI's default.
      expect(choice.detail).toContain("api.fireworks.ai");
    } finally {
      delete process.env.AGENTOPT_TEST_FW_KEY;
    }
  });

  it("errors when the named key env var is unset (and no OPENAI_API_KEY fallback)", async () => {
    delete process.env.AGENTOPT_TEST_ABSENT_KEY;
    const savedOpenai = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    try {
      await expect(
        createJudgeBackend({
          backend: "claude-code",
          judge_backend: "cloud",
          judge_base_url: "https://api.fireworks.ai/inference/v1",
          judge_api_key_env: "AGENTOPT_TEST_ABSENT_KEY",
        }),
      ).rejects.toThrow(BackendUnavailableError);
    } finally {
      if (savedOpenai !== undefined) process.env.OPENAI_API_KEY = savedOpenai;
    }
  });
});

describe("CachingJudge (verdict reuse)", () => {
  function counting(): { judge: Judge; calls: () => number } {
    let n = 0;
    const judge: Judge = {
      async evaluate() {
        n++;
        return { score: 0.7, reasoning: "graded" };
      },
    };
    return { judge, calls: () => n };
  }

  it("grades an unchanged (output, rubric) once, then serves from cache", async () => {
    const { judge, calls } = counting();
    const cached = new CachingJudge(judge);
    const input = { rubric: "polite", output: "hi there", inputText: null, good: [], bad: [], glossary: {} };
    const a = await cached.evaluate(input);
    const b = await cached.evaluate(input); // identical -> cache hit
    expect(calls()).toBe(1); // graded only once — this is the double-baseline saving
    expect(b).toEqual(a);
  });

  it("re-grades when the OUTPUT changes (code change -> new answer -> cache miss)", async () => {
    const { judge, calls } = counting();
    const cached = new CachingJudge(judge);
    const base = { rubric: "polite", inputText: null, good: [] as string[], bad: [] as string[], glossary: {} };
    await cached.evaluate({ ...base, output: "answer v1" });
    await cached.evaluate({ ...base, output: "answer v2" }); // a changed answer
    expect(calls()).toBe(2); // NEVER reuses a stale grade for a different output
  });

  it("re-grades when the rubric or anchors change (same output)", async () => {
    const { judge, calls } = counting();
    const cached = new CachingJudge(judge);
    const base = { output: "same answer", inputText: null, good: [] as string[], bad: [] as string[], glossary: {} };
    await cached.evaluate({ ...base, rubric: "be polite" });
    await cached.evaluate({ ...base, rubric: "be concise" }); // different rubric
    await cached.evaluate({ ...base, rubric: "be polite", good: ["hi!"] }); // different anchors
    expect(calls()).toBe(3);
  });
});

describe("searchGuidance", () => {
  it("fetches grounded guidance via claude, injecting the model and source contract", async () => {
    const backend = new ClaudeCodeBackend();
    const spy = vi
      .spyOn(backend, "complete")
      .mockResolvedValue("SEARCHED: use an output-format constraint — ... — https://docs.example");
    const out = await searchGuidance(backend, "GOAL(S) FAILING:\n- be concise", "kimi-k2p6");
    expect(out).toContain("SEARCHED:");
    const sent = spy.mock.calls[0]![0].map((m) => m.content).join("\n");
    expect(sent).toContain("OPEN and READ"); // instructs it to actually fetch sources
    expect(sent).toContain("SOURCE_URL"); // requires citations
    expect(sent).toContain('model "kimi-k2p6"'); // provider-aware
  });

  it("returns null for a backend that cannot search (no faking)", async () => {
    const out = await searchGuidance(new HttpBackend("cloud", "u", "k", "m"), "problem");
    expect(out).toBeNull();
  });
});

const ENTRY: ManifestEntry = {
  id: "prompt_agent_1",
  kind: "prompt",
  file: "agent.ts",
  line: 1,
  node_path: "TemplateLiteral@1:15",
  source_kind: "fstring",
  slots: ["name"],
  content_hash: null,
  editable: true,
  prompt_hashes: [],
  notes: null,
};

const FAILURE = {
  case_id: "rule_x",
  source: "Greet the fan by name.",
  layer: 2 as const,
  runs: 2,
  passed_runs: 0,
  status: "failed" as const,
  detail: "no greeting found",
};

describe("Reflector", () => {
  it("proposes a validated prompt candidate", async () => {
    const backend = new ScriptedBackend([
      'Sure! {"kind": "prompt", "new_content": "Hi {name}, welcome!", "description": "add greeting", "reasoning": "the rule asks for it"}',
    ]);
    const candidate = await new Reflector(backend).propose(
      [FAILURE],
      ENTRY,
      { text: "Hello {name}.", pattern: "Hello (.*?)\\." },
    );
    expect(candidate?.kind).toBe("prompt");
    expect(candidate?.new_content).toBe("Hi {name}, welcome!");
    expect(candidate?.tier).toBe(2);
    const sent = backend.prompts[0]!.map((m) => m.content).join("\n");
    expect(sent).toContain("Greet the fan by name.");
    expect(sent).toContain("SLOTS THAT MUST BE PRESERVED: {name}");
  });

  it("proposes a diverse population of candidates from a JSON array", async () => {
    const backend = new ScriptedBackend([
      JSON.stringify([
        { kind: "prompt", new_content: "Hi {name}!", description: "greet" },
        { kind: "params", param_overrides: { max_tokens: 512 }, description: "small bump" },
        { kind: "params", param_overrides: { max_tokens: 4096 }, description: "big bump" },
      ]),
    ]);
    const candidates = await new Reflector(backend).proposeCandidates(3, [FAILURE], ENTRY, {
      text: "Hello {name}.",
      pattern: null,
    });
    expect(candidates).toHaveLength(3);
    expect(candidates.map((c) => c.kind)).toEqual(["prompt", "params", "params"]);
    expect(candidates[2]!.param_overrides).toEqual({ max_tokens: 4096 });
    const sent = backend.prompts[0]!.map((m) => m.content).join("\n");
    expect(sent).toContain("GENUINELY DIFFERENT"); // diversity instruction present
  });

  it("dedupes identical candidates and falls back to a single proposal", async () => {
    const dupBackend = new ScriptedBackend([
      JSON.stringify([
        { kind: "params", param_overrides: { max_tokens: 4096 }, description: "a" },
        { kind: "params", param_overrides: { max_tokens: 4096 }, description: "dup" },
      ]),
    ]);
    const deduped = await new Reflector(dupBackend).proposeCandidates(3, [FAILURE], ENTRY, {
      text: "t",
      pattern: null,
    });
    expect(deduped).toHaveLength(1); // identical override collapsed

    // Not an array at all -> falls back to the single-candidate path.
    const single = new ScriptedBackend([
      '{"kind": "prompt", "new_content": "Hi {name}.", "description": "d"}',
    ]);
    const one = await new Reflector(single).proposeCandidates(3, [FAILURE], ENTRY, {
      text: "Hello {name}.",
      pattern: null,
    });
    expect(one).toHaveLength(1);
    expect(one[0]!.new_content).toBe("Hi {name}.");
  });

  it("steers to a params change once prompt rewrites keep failing", async () => {
    const backend = new ScriptedBackend([
      '{"kind": "params", "param_overrides": {"temperature": 0.3}, "description": "cool it"}',
    ]);
    const promptFail = (id: string) => ({
      id, ts: "2026-01-01T00:00:00Z", tier: 2, target_id: ENTRY.id,
      description: `prompt tweak ${id}`, outcome: "discarded" as const, reason: "no gain",
    });
    const candidate = await new Reflector(backend).propose(
      [FAILURE],
      ENTRY,
      { text: "Hello {name}.", pattern: null },
      [promptFail("a"), promptFail("b")], // two failed prompt edits -> switch lever
    );
    expect(candidate?.kind).toBe("params");
    const sent = backend.prompts[0]!.map((m) => m.content).join("\n");
    expect(sent).toContain("STRATEGY");
    expect(sent).toContain("do NOT propose another prompt edit");
  });

  it("surfaces runtime diagnostics and can propose a max_tokens change", async () => {
    const backend = new ScriptedBackend([
      '{"kind": "params", "param_overrides": {"max_tokens": 2048}, "description": "give the router room to answer"}',
    ]);
    const candidate = await new Reflector(backend).propose(
      [FAILURE],
      ENTRY,
      { text: "Classify the intent.", pattern: null },
      [],
      null,
      [],
      "the model returned EMPTY output on at least one failing case while max_tokens was 5",
    );
    expect(candidate?.kind).toBe("params");
    expect(candidate?.param_overrides).toEqual({ max_tokens: 2048 });
    const sent = backend.prompts[0]!.map((m) => m.content).join("\n");
    expect(sent).toContain("RUNTIME DIAGNOSTICS");
    expect(sent).toContain("EMPTY output");
    expect(sent).toContain("max_tokens"); // vocabulary now names it
  });

  it("retries once on a slot violation, then gives up", async () => {
    const dropped = '{"kind": "prompt", "new_content": "no slot", "description": "d"}';
    const backendRecovers = new ScriptedBackend([
      dropped,
      '{"kind": "prompt", "new_content": "fixed {name}", "description": "d"}',
    ]);
    const recovered = await new Reflector(backendRecovers).propose([FAILURE], ENTRY, {
      text: "Hello {name}.",
      pattern: null,
    });
    expect(recovered?.new_content).toBe("fixed {name}");
    expect(backendRecovers.prompts[1]!.at(-1)?.content).toContain("missing slots");

    const backendFails = new ScriptedBackend([dropped, dropped]);
    expect(
      await new Reflector(backendFails).propose([FAILURE], ENTRY, { text: "t", pattern: null }),
    ).toBeNull();
  });

  it("includes past discarded attempts and distills lessons", async () => {
    const backend = new ScriptedBackend([
      '{"kind": "params", "param_overrides": {"temperature": 0.1}, "description": "cool it"}',
      "Always greet fans by their first name in the opening sentence.",
    ]);
    const reflector = new Reflector(backend);
    const candidate = await reflector.propose(
      [FAILURE],
      ENTRY,
      { text: "t {name}", pattern: null },
      [
        {
          id: "att_1",
          ts: "2026-01-01T00:00:00Z",
          tier: 2,
          target_id: ENTRY.id,
          description: "make it shouty",
          outcome: "discarded",
          reason: "no overall gain",
        },
      ],
    );
    expect(candidate?.kind).toBe("params");
    expect(candidate?.tier).toBe(1);
    expect(backend.prompts[0]!.map((m) => m.content).join("\n")).toContain(
      "make it shouty -> no overall gain",
    );

    const lesson = await reflector.distillLesson(candidate!, [FAILURE]);
    expect(lesson).toContain("greet fans");
    // A params fix is distilled with a params-shaped label, not "NEW PROMPT".
    const distillPrompt = backend.prompts[1]!.at(-1)?.content ?? "";
    expect(distillPrompt).toContain("PARAM CHANGE");
    expect(distillPrompt).toContain("temperature");
  });

  it("grounds a code lesson in the diff it is given, and strips preamble", async () => {
    const backend = new ScriptedBackend([
      "A future engineer or agent should remember:\n\nAudit the output pipeline: a post-processing step was stripping required formatting.",
    ]);
    const codeCandidate: Candidate = {
      id: "c",
      tier: 4,
      entry_id: "e",
      kind: "code",
      description: "delegated code fix (llm.ts)",
      reasoning: "",
      new_content: null,
      param_overrides: null,
      code_instructions: "removed the stripping step",
    };
    const diff = "CODE DIFF:\n-  return format(text)\n+  return text";
    const lesson = await new Reflector(backend).distillLesson(codeCandidate, [FAILURE], diff);
    // The provided diff reaches the model...
    expect(backend.prompts[0]!.at(-1)?.content).toContain("return format(text)");
    // ...and the label preamble is stripped from the returned lesson.
    expect(lesson).toBe(
      "Audit the output pipeline: a post-processing step was stripping required formatting.",
    );
  });

  it("stripLessonPreamble removes labels, wrapper paragraphs, and quotes", () => {
    expect(stripLessonPreamble("Plain sentence.")).toBe("Plain sentence.");
    expect(stripLessonPreamble('"Quoted sentence."')).toBe("Quoted sentence.");
    expect(stripLessonPreamble("Lesson: Raise max_tokens for reasoning models.")).toBe(
      "Raise max_tokens for reasoning models.",
    );
    expect(stripLessonPreamble("Remember this:\n\nDon't strip brackets.")).toBe(
      "Don't strip brackets.",
    );
  });

  it("frames what already worked so the next change builds on it", async () => {
    const backend = new ScriptedBackend([
      '{"kind": "prompt", "new_content": "extend {name}", "description": "next step"}',
    ]);
    await new Reflector(backend).propose(
      [FAILURE],
      ENTRY,
      { text: "Hello {name}.", pattern: null },
      [], // no discarded attempts
      null,
      [
        {
          id: "att_ok",
          ts: "2026-01-01T00:00:00Z",
          tier: 2,
          target_id: ENTRY.id,
          description: "lead with the answer, no headers",
          outcome: "applied",
          score_delta: 38,
          version: "v1",
        },
      ],
    );
    const sent = backend.prompts[0]!.map((m) => m.content).join("\n");
    expect(sent).toContain("WHAT ALREADY WORKED");
    expect(sent).toContain("v1: lead with the answer, no headers (+38 pts)");
  });
});
