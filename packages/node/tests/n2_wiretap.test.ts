/**
 * N2: the fetch wiretap against real request shapes — OpenAI format,
 * Anthropic format, AI-SDK-style streaming — with a stubbed network.
 */

import { mkdtempSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { promptHash } from "../src/hashing.js";
import { parseTraceLine, type LlmCall } from "../src/models/trace.js";
import { startWiretap } from "../src/wiretap/session.js";
import { applyOverlayToBody } from "../src/wiretap/overlay.js";
import { retryPolicy } from "../src/wiretap/fetchpatch.js";

const OPENAI_RESPONSE = {
  id: "chatcmpl-1",
  object: "chat.completion",
  model: "kimi-k2.5",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "intent: refund" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 42, completion_tokens: 5, total_tokens: 47 },
};

const ANTHROPIC_RESPONSE = {
  id: "msg-1",
  type: "message",
  content: [{ type: "text", text: "hello there" }],
  usage: { input_tokens: 10, output_tokens: 3 },
};

function stubFetch(payload: unknown, status = 200, headers?: Record<string, string>) {
  return vi.fn(async () =>
    new Response(
      typeof payload === "string" ? payload : JSON.stringify(payload),
      { status, headers: headers ?? { "content-type": "application/json" } },
    ),
  );
}

function readEvents(tracePath: string) {
  return readFileSync(tracePath, "utf-8")
    .trim()
    .split("\n")
    .map(parseTraceLine);
}

function project(): string {
  return mkdtempSync(join(tmpdir(), "agentopt-"));
}

const realFetch = globalThis.fetch;
const realRetryDelay = retryPolicy.baseDelayMs;
beforeEach(() => {
  retryPolicy.baseDelayMs = 0; // no real backoff sleeps in tests
});
afterEach(() => {
  globalThis.fetch = realFetch;
  retryPolicy.baseDelayMs = realRetryDelay;
});

describe("fetch wiretap", () => {
  it("captures an openai-format call with call site and hash", async () => {
    const stub = stubFetch(OPENAI_RESPONSE);
    globalThis.fetch = stub as unknown as typeof fetch;
    const root = project();

    const run = startWiretap({ projectRoot: root, inputText: "money back pls??" });
    try {
      await fetch("https://api.fireworks.ai/inference/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({
          model: "kimi-k2.5",
          temperature: 0.3,
          max_tokens: 512,
          messages: [
            { role: "system", content: "You classify fan messages into intents." },
            { role: "user", content: "money back pls??" },
          ],
        }),
      });
    } finally {
      await run.stop();
    }

    const events = readEvents(run.tracePath);
    expect(events[0]?.kind).toBe("run_meta");
    const call = events[1] as LlmCall;
    expect(call.kind).toBe("llm_call");
    expect(call.provider).toBe("fireworks");
    expect(call.model).toBe("kimi-k2.5");
    expect(call.params?.temperature).toBe(0.3);
    expect(call.params?.max_output_tokens).toBe(512);
    expect(call.response?.content).toBe("intent: refund");
    expect(call.usage?.input_tokens).toBe(42);
    expect(call.prompt_hash).toBe(
      promptHash([
        { role: "system", content: "You classify fan messages into intents." },
        { role: "user", content: "money back pls??" },
      ]),
    );
    expect(call.call_site?.file).toContain("n2_wiretap.test.ts");
  });

  it("captures anthropic format with system as leading message", async () => {
    globalThis.fetch = stubFetch(ANTHROPIC_RESPONSE) as unknown as typeof fetch;
    const root = project();

    const run = startWiretap({ projectRoot: root });
    try {
      await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        body: JSON.stringify({
          model: "claude-haiku-4-5",
          max_tokens: 100,
          system: "Be brief.",
          messages: [{ role: "user", content: "hi" }],
        }),
      });
    } finally {
      await run.stop();
    }

    const call = readEvents(run.tracePath)[1] as LlmCall;
    expect(call.provider).toBe("anthropic");
    expect(call.messages[0]).toEqual({ role: "system", content: "Be brief." });
    expect(call.response?.content).toBe("hello there");
    expect(call.usage?.output_tokens).toBe(3);
  });

  it("reassembles streamed SSE replies", async () => {
    const sse = [
      'data: {"choices":[{"delta":{"content":"The premium "}}]}',
      'data: {"choices":[{"delta":{"content":"plan is $19.99."},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":6}}',
      "data: [DONE]",
      "",
    ].join("\n");
    globalThis.fetch = stubFetch(sse, 200, {
      "content-type": "text/event-stream",
    }) as unknown as typeof fetch;
    const root = project();

    const run = startWiretap({ projectRoot: root });
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({
          model: "gpt-4o-mini",
          stream: true,
          messages: [{ role: "user", content: "price?" }],
        }),
      });
      // the caller can still read the stream untouched
      expect(await response.text()).toContain("[DONE]");
    } finally {
      await run.stop();
    }

    const call = readEvents(run.tracePath)[1] as LlmCall;
    expect(call.response?.content).toBe("The premium plan is $19.99.");
    expect(call.usage?.input_tokens).toBe(7);
  });

  it("records provider errors and passes them through unchanged", async () => {
    globalThis.fetch = stubFetch({ error: "boom" }, 500) as unknown as typeof fetch;
    const root = project();

    const run = startWiretap({ projectRoot: root });
    let status = 0;
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "x" }] }),
      });
      status = response.status;
    } finally {
      await run.stop();
    }

    expect(status).toBe(500);
    const call = readEvents(run.tracePath)[1] as LlmCall;
    expect(call.error).toBe("HTTP 500");
    expect(call.response).toBeFalsy();
  });

  it("retries a transient failure so the agent sees the eventual success", async () => {
    let calls = 0;
    globalThis.fetch = vi.fn(async () => {
      calls += 1;
      if (calls < 3) return new Response(JSON.stringify({ error: "overloaded" }), { status: 503 });
      return new Response(JSON.stringify(OPENAI_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch;
    const root = project();
    const run = startWiretap({ projectRoot: root });
    let status = 0;
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "x" }] }),
      });
      status = response.status;
    } finally {
      await run.stop();
    }
    expect(calls).toBe(3); // two 503s retried, then success
    expect(status).toBe(200); // the agent never saw the transient failures
    const call = readEvents(run.tracePath)[1] as LlmCall;
    expect(call.response?.content).toBe("intent: refund");
  });

  it("retries a network drop, then succeeds", async () => {
    let calls = 0;
    globalThis.fetch = vi.fn(async () => {
      calls += 1;
      if (calls === 1) throw Object.assign(new TypeError("fetch failed"), { cause: { code: "ECONNRESET" } });
      return new Response(JSON.stringify(OPENAI_RESPONSE), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch;
    const root = project();
    const run = startWiretap({ projectRoot: root });
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "x" }] }),
      });
      expect(response.status).toBe(200);
    } finally {
      await run.stop();
    }
    expect(calls).toBe(2);
  });

  it("does not retry a non-transient 4xx", async () => {
    let calls = 0;
    globalThis.fetch = vi.fn(async () => {
      calls += 1;
      return new Response("bad request", { status: 400 });
    }) as unknown as typeof fetch;
    const root = project();
    const run = startWiretap({ projectRoot: root });
    let status = 0;
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "x" }] }),
      });
      status = response.status;
    } finally {
      await run.stop();
    }
    expect(calls).toBe(1); // 400 is the agent's problem, not transient — no retry
    expect(status).toBe(400);
  });

  it("gives up after the retry budget and passes the failure through", async () => {
    let calls = 0;
    globalThis.fetch = vi.fn(async () => {
      calls += 1;
      return new Response("busy", { status: 503 });
    }) as unknown as typeof fetch;
    const root = project();
    const run = startWiretap({ projectRoot: root });
    let status = 0;
    try {
      const response = await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({ model: "m", messages: [{ role: "user", content: "x" }] }),
      });
      status = response.status;
    } finally {
      await run.stop();
    }
    expect(calls).toBe(1 + retryPolicy.maxRetries); // first attempt + retries
    expect(status).toBe(503); // a persistent failure still surfaces to the agent
  });

  it("ignores non-LLM requests and captures nothing outside a run", async () => {
    const stub = stubFetch({ ok: true });
    globalThis.fetch = stub as unknown as typeof fetch;
    const root = project();

    const run = startWiretap({ projectRoot: root });
    try {
      await fetch("https://example.com/api/users", {
        method: "POST",
        body: JSON.stringify({ hello: 1 }),
      });
    } finally {
      await run.stop();
    }
    await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      body: JSON.stringify({ model: "m", messages: [] }),
    });

    expect(readEvents(run.tracePath)).toHaveLength(1); // run_meta only
  });
});

describe("overlay", () => {
  it("rewrites matching content and merges params", () => {
    const body = applyOverlayToBody(
      {
        rules: [
          {
            entry_id: "e",
            kind: "literal",
            match_exact: "old prompt",
            new_template: "new prompt",
            params: { temperature: 0.1 },
            slots: [],
          },
        ],
      },
      {
        messages: [
          { role: "system", content: "old prompt" },
          { role: "user", content: "hi" },
        ],
        temperature: 0.9,
      },
    );
    expect((body.messages as Array<{ content: string }>)[0]?.content).toBe("new prompt");
    expect(body.temperature).toBe(0.1);
  });

  it("re-renders template slots from live values", () => {
    const body = applyOverlayToBody(
      {
        rules: [
          {
            entry_id: "e",
            kind: "fstring",
            match_pattern: "You help (.*?) with (.*?)\\.",
            slots: ["name", "topic"],
            new_template: "You warmly help {name}, expert in {topic}.",
          },
        ],
      },
      { messages: [{ role: "system", content: "You help Ana with billing." }] },
    );
    expect((body.messages as Array<{ content: string }>)[0]?.content).toBe(
      "You warmly help Ana, expert in billing.",
    );
  });

  it("applies model_override and is inert when unmatched", () => {
    const body = applyOverlayToBody(
      { rules: [], model_override: "other-model" },
      { model: "orig", messages: [{ role: "user", content: "untouched" }] },
    );
    expect(body.model).toBe("other-model");
    expect((body.messages as Array<{ content: string }>)[0]?.content).toBe("untouched");
  });

  it("overlay is applied to the outgoing request during a run", async () => {
    let sentBody = "";
    globalThis.fetch = vi.fn(async (_input: unknown, init?: RequestInit) => {
      sentBody = String(init?.body ?? "");
      return new Response(JSON.stringify(OPENAI_RESPONSE), {
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch;
    const root = project();

    const run = startWiretap({
      projectRoot: root,
      mode: "eval",
      overlay: {
        rules: [
          {
            entry_id: "e",
            kind: "literal",
            match_exact: "You are terse.",
            new_template: "You are terse. Premium is $19.99.",
            slots: [],
          },
        ],
      },
    });
    try {
      await fetch("https://api.openai.com/v1/chat/completions", {
        method: "POST",
        body: JSON.stringify({
          model: "m",
          messages: [{ role: "system", content: "You are terse." }],
        }),
      });
    } finally {
      await run.stop();
    }

    expect(sentBody).toContain("Premium is $19.99.");
    const call = readEvents(run.tracePath)[1] as LlmCall;
    expect(call.messages[0]?.content).toContain("$19.99"); // trace shows the trialed prompt
  });
});
