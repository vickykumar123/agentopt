/**
 * N12: attach mode. The user runs their own server with the wiretap
 * registered (registerServer); agentopt does NOT boot it. It sends the trial
 * overlay and a per-request run id as headers, and reads this request's calls
 * out of the shared attach trace — no restart.
 */

import { execSync, spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { loadConfig } from "../src/models/config.js";
import { runnerFor } from "../src/runner.js";
import type { LlmCall } from "../src/models/trace.js";
import { readFileSync } from "node:fs";
import { tempProject, freePort } from "./helpers.js";

const packageRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

/** A user server that registers the wiretap itself, then serves /chat. */
function serverSource(distIndex: string): string {
  return `import { registerServer } from ${JSON.stringify(distIndex)};
import { createServer } from "node:http";

registerServer({ projectRoot: process.cwd() });

const SYSTEM = "You are a terse fitness coach.";
const PORT = Number(process.env.PORT);

function reply(system) { return system.includes("Reply in French") ? "bonjour" : "hi there"; }

// Read the body via an awaited promise, like a real framework (await req.json()):
// the LLM call then runs in the request's async context, preserving it.
function readBody(req) {
  return new Promise((resolve) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => resolve(raw));
  });
}

const server = createServer(async (req, res) => {
  if (req.url === "/health") { res.end("ok"); return; }
  const body = JSON.parse((await readBody(req)) || "{}");
  if (req.url.startsWith("/v1/chat/completions")) {
    const system = body.messages.find((m) => m.role === "system").content;
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify({
      model: "fake",
      choices: [{ index: 0, message: { role: "assistant", content: reply(system) }, finish_reason: "stop" }],
      usage: { prompt_tokens: 10, completion_tokens: 2 },
    }));
    return;
  }
  // /chat — call the (self-hosted) model through fetch, captured by the wiretap
  const upstream = await fetch(\`http://127.0.0.1:\${PORT}/v1/chat/completions\`, {
    method: "POST",
    body: JSON.stringify({ model: "fake", messages: [
      { role: "system", content: SYSTEM },
      { role: "user", content: body.message },
    ] }),
  });
  const data = await upstream.json();
  res.setHeader("content-type", "application/json");
  res.end(JSON.stringify({ reply: data.choices[0].message.content }));
});
server.listen(PORT, "127.0.0.1", () => console.log("ready"));
`;
}

let child: ChildProcess | null = null;
afterEach(() => {
  child?.kill();
  child = null;
});

describe("http attach mode", () => {
  it("attaches to a user-run server and trials via per-request headers", async () => {
    execSync("npm run build", { cwd: packageRoot, stdio: "pipe", timeout: 180000 });
    const distIndex = pathToFileURL(join(packageRoot, "dist", "index.js")).href;

    const port = await freePort();
    const root = tempProject({
      "server.mjs": serverSource(distIndex),
      "agentopt.yaml":
        // no `start:` → attach mode
        "run:\n" +
        "  http:\n" +
        `    ready: http://127.0.0.1:${port}/health\n` +
        `    url: http://127.0.0.1:${port}/chat\n` +
        '    body: \'{"message":"{input}"}\'\n' +
        "    output: $.reply\n",
    });

    // the USER starts the server (agentopt does not)
    child = spawn(process.execPath, ["server.mjs"], {
      cwd: root,
      env: { ...process.env, PORT: String(port) },
      stdio: "ignore",
    });
    await waitFor(`http://127.0.0.1:${port}/health`);

    const runner = runnerFor(loadConfig(join(root, "agentopt.yaml")), root);
    try {
      // 1. plain request — captured into an isolated per-run trace by run id
      const plain = await runner.run("what's a good warm-up?", "eval");
      expect(plain.output).toBe("hi there");
      expect(existsSync(plain.tracePath)).toBe(true);
      const events = readFileSync(plain.tracePath, "utf-8")
        .split("\n")
        .filter(Boolean)
        .map((line) => JSON.parse(line) as LlmCall);
      const call = events.find((e) => e.kind === "llm_call")!;
      expect(call.messages[0]?.content).toBe("You are a terse fitness coach.");
      expect(call.run_id).toBe(plain.runId); // stamped with this request's id

      // 2. a per-request overlay header rewrites the prompt — no server restart
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
      // and its trace is isolated from the plain run's
      expect(trialed.runId).not.toBe(plain.runId);
    } finally {
      await runner.close?.();
    }
  }, 240000);
});

async function waitFor(url: string): Promise<void> {
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      await fetch(url);
      return;
    } catch {
      await new Promise((r) => setTimeout(r, 100));
    }
  }
  throw new Error(`server never came up at ${url}`);
}
