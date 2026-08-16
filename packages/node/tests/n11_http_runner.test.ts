/**
 * N11: the `run.http` harness. Unit tests for output extraction and env/input
 * interpolation, plus a full integration run against a real subprocess HTTP
 * server booted with the wiretap hook — request → response → trace → overlay.
 */

import { execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadConfig } from "../src/models/config.js";
import { extractPath, HttpRunner, runnerFor } from "../src/runner.js";
import type { LlmCall } from "../src/models/trace.js";
import { tempProject, freePort } from "./helpers.js";

describe("output extraction", () => {
  const body = JSON.stringify({
    reply: "hello there",
    choices: [{ message: { content: "nested text" } }],
  });
  it("reads dot and bracket paths, JSONPath `$.` prefix optional", () => {
    expect(extractPath(body, "$.reply")).toBe("hello there");
    expect(extractPath(body, "reply")).toBe("hello there");
    expect(extractPath(body, "$.choices[0].message.content")).toBe("nested text");
  });
  it("returns null for non-JSON or a missing path (→ structured invalid_response, not an empty answer)", () => {
    expect(extractPath("plain text", "$.reply")).toBeNull();
    expect(extractPath(body, "$.missing.deep")).toBeNull();
  });
  it("serializes non-string leaves", () => {
    expect(extractPath(JSON.stringify({ n: 42 }), "n")).toBe("42");
  });
});

describe("config", () => {
  it("accepts an http target with headers/body/start and rejects extra keys", () => {
    const root = tempProject({
      "agentopt.yaml":
        "run:\n" +
        "  http:\n" +
        "    start: node server.mjs\n" +
        "    url: http://localhost:8123/chat\n" +
        "    headers:\n" +
        '      authorization: "Bearer ${TOK}"\n' +
        '    body: \'{"message":"{input}"}\'\n' +
        "    output: $.reply\n",
    });
    const config = loadConfig(join(root, "agentopt.yaml"));
    expect(config.run.http?.url).toBe("http://localhost:8123/chat");
    expect(config.run.http?.method).toBe("POST"); // default
    expect(config.run.http?.headers.authorization).toBe("Bearer ${TOK}");
    expect(runnerFor(config, root)).toBeInstanceOf(HttpRunner);
  });

  it("still enforces exactly-one run style", () => {
    const root = tempProject({
      "agentopt.yaml": "run:\n  node: a.mjs:run\n  http:\n    url: http://x/y\n",
    });
    expect(() => loadConfig(join(root, "agentopt.yaml"))).toThrow(/exactly one/);
  });
});

// A self-contained "agent server": reads {message} from the POST body, calls a
// (fake) OpenAI endpoint it also hosts, and returns the model's reply. The
// system prompt is a literal so the wiretap/overlay can act on it.
const AGENT_SERVER = `import { createServer } from "node:http";

const SYSTEM = "You are a terse fitness coach.";

function llmServer() {
  return createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      const body = JSON.parse(raw || "{}");
      const system = body.messages.find((m) => m.role === "system").content;
      const reply = system.includes("Reply in French") ? "bonjour" : "hi there";
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        model: "fake", choices: [{ message: { role: "assistant", content: reply }, finish_reason: "stop" }],
        usage: { prompt_tokens: 10, completion_tokens: 2 },
      }));
    });
  });
}

const llm = llmServer();
await new Promise((r) => llm.listen(0, "127.0.0.1", r));
const llmPort = llm.address().port;

const app = createServer((req, res) => {
  if (req.url === "/health") { res.end("ok"); return; }
  let raw = "";
  req.on("data", (c) => (raw += c));
  req.on("end", async () => {
    const { message } = JSON.parse(raw || "{}");
    const upstream = await fetch(\`http://127.0.0.1:\${llmPort}/v1/chat/completions\`, {
      method: "POST",
      body: JSON.stringify({ model: "fake", messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: message },
      ] }),
    });
    const data = await upstream.json();
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({ reply: data.choices[0].message.content }));
  });
});
const port = Number(process.env.PORT ?? 0);
await new Promise((r) => app.listen(port, "127.0.0.1", r));
console.log("PORT=" + app.address().port);
`;

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
  vi.restoreAllMocks();
});

describe("http harness integration", () => {
  it("boots the server with the hook, sends requests, captures traces, applies overlays", async () => {
    const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
    execSync("npm run build", { cwd: packageRoot, stdio: "pipe", timeout: 180000 });
    expect(existsSync(join(packageRoot, "dist", "hook.js"))).toBe(true);

    const port = await freePort();
    const root = tempProject({
      "server.mjs": AGENT_SERVER,
      "agentopt.yaml":
        "run:\n" +
        "  http:\n" +
        `    start: node server.mjs\n` +
        `    ready: http://127.0.0.1:${port}/health\n` +
        `    url: http://127.0.0.1:${port}/chat\n` +
        '    body: \'{"message":"{input}"}\'\n' +
        "    output: $.reply\n",
    });
    // the server reads PORT from env; inject it via the process env the runner inherits
    process.env.PORT = String(port);

    const config = loadConfig(join(root, "agentopt.yaml"));
    const runner = runnerFor(config, root);
    try {
      // 1. plain run — output extracted from the JSON response
      const plain = await runner.run("what's a good warm-up?", "eval");
      expect(plain.exitCode).toBe(0);
      expect(plain.output).toBe("hi there");
      // 2. the LLM call was captured into an isolated per-run trace
      expect(existsSync(plain.tracePath)).toBe(true);
      const events = readTrace(plain.tracePath);
      const call = events.find((e) => e.kind === "llm_call") as LlmCall;
      expect(call.messages[0]?.content).toBe("You are a terse fitness coach.");
      expect(call.response?.content).toBe("hi there");

      // 3. an overlay restarts the server and rewrites the outgoing prompt
      const trialed = await runner.run("what's a good warm-up?", "eval", {
        rules: [
          {
            entry_id: "e",
            kind: "literal",
            match_exact: "You are a terse fitness coach.",
            new_template: "You are a terse fitness coach. Reply in French.",
            slots: [],
          },
        ],
      });
      expect(trialed.output).toBe("bonjour");
    } finally {
      await runner.close?.();
      delete process.env.PORT;
    }
  }, 240000);

  it("treats a non-2xx response as an agent failure, not a valid answer", async () => {
    const port = await freePort();
    const root = tempProject({
      // A server that always errors with a plausible-looking fallback body — the
      // exact trap: without a status check this reads as the agent's answer.
      "server.mjs":
        'import { createServer } from "node:http";\n' +
        'const app = createServer((req, res) => {\n' +
        '  if (req.url === "/health") { res.end("ok"); return; }\n' +
        '  res.statusCode = 500;\n' +
        '  res.setHeader("content-type", "application/json");\n' +
        '  res.end(JSON.stringify({ reply: "fallback" }));\n' +
        "});\n" +
        `app.listen(${port}, "127.0.0.1");\n`,
      "agentopt.yaml":
        "run:\n" +
        "  http:\n" +
        `    start: node server.mjs\n` +
        `    ready: http://127.0.0.1:${port}/health\n` +
        `    url: http://127.0.0.1:${port}/chat\n` +
        '    body: \'{"message":"{input}"}\'\n' +
        "    output: $.reply\n",
    });
    const config = loadConfig(join(root, "agentopt.yaml"));
    const runner = runnerFor(config, root);
    try {
      const result = await runner.run("hello", "eval");
      expect(result.exitCode).toBe(1);
      expect(result.output).toContain("HTTP 500");
      expect(result.output).not.toBe("fallback"); // the 500 body was NOT accepted
    } finally {
      await runner.close?.();
    }
  }, 60000);
});

function readTrace(path: string) {
  return readFileSync(path, "utf-8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

describe("httpFailureCategory", () => {
  it("classifies timeouts, provider errors, and runner errors", async () => {
    const { httpFailureCategory, RunnerError } = await import("../src/runner.js");
    expect(httpFailureCategory(new RunnerError("agent server returned HTTP 500: x"))).toBe("provider_error");
    expect(httpFailureCategory(new TypeError("fetch failed"))).toBe("provider_error");
    const abort = new Error("aborted"); abort.name = "AbortError";
    expect(httpFailureCategory(abort)).toBe("timeout");
    expect(httpFailureCategory(new Error("weird"))).toBe("runner_error");
  });
});
