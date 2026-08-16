/**
 * N13: the universal seams. `instrument()` captures a model call from a
 * provider the auto-wiretap does NOT patch (a plain function — no fetch, no
 * openai SDK); the captured prompt is located in source and trialed via
 * overlay exactly like an auto-captured call. `expose()` normalizes output.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { declaredSurfaceNames, expose, instrument } from "../src/instrument.js";
import { buildManifest } from "../src/manifest/builder.js";
import { PromptLocator } from "../src/manifest/locator.js";
import { parseTraceLine, type LlmCall, type Message } from "../src/models/trace.js";
import type { Overlay } from "../src/wiretap/overlay.js";
import { startWiretap } from "../src/wiretap/session.js";
import { tempProject } from "./helpers.js";

const SYSTEM = "You are a terse fitness coach.";

// A made-up provider with its own transport — nothing agentopt auto-patches.
function fakeProvider(messages: Message[]): string {
  const system = messages.find((m) => m.role === "system")?.content ?? "";
  return system.includes("Reply in French") ? "bonjour" : "hi there";
}

function readEvents(tracePath: string) {
  return readFileSync(tracePath, "utf-8").trim().split("\n").map(parseTraceLine);
}

async function runOnce(root: string, overlay: Overlay | null) {
  const run = startWiretap({
    projectRoot: root,
    mode: overlay ? "eval" : "observe",
    inputText: "warm up please",
    overlay,
  });
  const output = await instrument(
    {
      model: "fake-1",
      provider: "custom",
      params: { temperature: 0.2 },
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: "warm up please" },
      ],
    },
    (messages) => fakeProvider(messages),
  );
  await run.stop();
  return { output, events: readEvents(run.tracePath) };
}

describe("instrument + expose", () => {
  it("captures a non-fetch provider call, locates the prompt, applies an overlay", async () => {
    // the prompt literal lives in the project's source, as a real agent's would
    const root = tempProject({ "agent.mjs": `export const SYSTEM = ${JSON.stringify(SYSTEM)};\n` });

    // 1. plain run — instrument captured the call the wiretap never sees
    const plain = await runOnce(root, null);
    expect(plain.output).toBe("hi there");
    const call = plain.events.find((e) => e.kind === "llm_call") as LlmCall;
    expect(call.provider).toBe("custom");
    expect(call.model).toBe("fake-1");
    expect(call.params?.temperature).toBe(0.2);
    expect(call.messages[0]?.content).toBe(SYSTEM);
    expect(call.response?.content).toBe("hi there");
    expect(call.prompt_hash).toMatch(/^[0-9a-f]{64}$/);

    // 2. the captured prompt is locatable in source and marked editable
    const manifest = buildManifest(root, plain.events);
    const entry = manifest.entries.find((e) => e.editable && e.kind === "prompt");
    expect(entry?.file).toBe("agent.mjs");
    expect(new PromptLocator(root).find(SYSTEM, "agent.mjs")).toHaveLength(1);

    // 3. an overlay rewrites the outgoing prompt — the provider sees it
    const trialed = await runOnce(root, {
      rules: [
        {
          entry_id: "e",
          kind: "literal",
          match_exact: SYSTEM,
          new_template: `${SYSTEM} Reply in French.`,
          slots: [],
        },
      ],
    });
    expect(trialed.output).toBe("bonjour");
    const trialedCall = trialed.events.find((e) => e.kind === "llm_call") as LlmCall;
    expect(trialedCall.messages[0]?.content).toBe(`${SYSTEM} Reply in French.`);
  });

  it("is a passthrough outside an agentopt run (production)", async () => {
    const result = await instrument(
      { model: "m", messages: [{ role: "user", content: "hi" }] },
      (messages) => `saw ${messages.length} messages`,
    );
    expect(result).toBe("saw 1 messages");
  });

  it("captures via a custom response mapper (structured provider result)", async () => {
    const root = tempProject();
    const run = startWiretap({ projectRoot: root, mode: "observe" });
    await instrument(
      {
        model: "m",
        messages: [{ role: "user", content: "hi" }],
        response: (result) => ({
          content: (result as { text: string }).text,
          usage: { input_tokens: 11, output_tokens: 3 },
        }),
      },
      () => ({ text: "mapped reply", raw: {} }),
    );
    await run.stop();
    const call = readEvents(run.tracePath).find((e) => e.kind === "llm_call") as LlmCall;
    expect(call.response?.content).toBe("mapped reply");
    expect(call.usage?.input_tokens).toBe(11);
  });

  it("expose coerces any return to a string", async () => {
    expect(await expose(() => 42)(null)).toBe("42");
    expect(await expose(() => null)(null)).toBe("");
    expect(await expose(async () => ({ toString: () => "obj" }))(null)).toBe("obj");
  });

  it("expose function form passes the surface as a second argument", async () => {
    const run = expose((input, surface) => `${surface ?? "default"}:${input}`);
    expect(await run("x", "router")).toBe("router:x");
    expect(await run("x")).toBe("default:x");
  });

  it("expose map form dispatches on surface and declares the surface set", async () => {
    const run = expose({
      default: (input) => `coach:${input}`,
      router: (input) => `route:${input}`,
    });
    expect(await run("hi", "router")).toBe("route:hi");
    expect(await run("hi")).toBe("coach:hi"); // undefined surface -> default
    expect(await run("hi", "unknown")).toBe("coach:hi"); // unknown -> default
    expect([...declaredSurfaceNames()].sort()).toEqual(["default", "router"]);
  });

  it("expose map form throws when no handler and no default", async () => {
    const run = expose({ router: (input) => `route:${input}` });
    await expect(run("hi")).rejects.toThrow(/no "default" handler/);
  });
});
