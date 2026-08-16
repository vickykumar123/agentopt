/** Shared test fixtures: temp projects, a scripted LLM "network". */

import { createServer, type AddressInfo } from "node:net";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { vi } from "vitest";
import type { ChatMessage, LlmBackend } from "../src/gateway.js";

/** An OS-assigned free TCP port (bind to 0, read the assignment, release it). Avoids
 * the collision-prone fixed port ranges that made HTTP tests flaky under load. */
export function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.on("error", reject);
    srv.listen(0, "127.0.0.1", () => {
      const port = (srv.address() as AddressInfo).port;
      srv.close(() => resolve(port));
    });
  });
}

export function tempProject(files: Record<string, string> = {}): string {
  const root = mkdtempSync(join(tmpdir(), "agentopt-node-"));
  for (const [relpath, content] of Object.entries(files)) {
    const target = join(root, relpath);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, content, "utf-8");
  }
  return root;
}

/**
 * A fake OpenAI-format endpoint: `reply` sees the outgoing system and user
 * text (post-overlay, since the wiretap rewrites before fetch) and returns
 * the assistant text. Installs onto globalThis.fetch; caller restores.
 */
export function installFakeLlm(
  reply: (system: string, user: string) => string,
): ReturnType<typeof vi.fn> {
  const fake = vi.fn(async (_input: unknown, init?: RequestInit) => {
    const body = JSON.parse(String(init?.body ?? "{}")) as {
      messages?: Array<{ role: string; content: string }>;
    };
    const messages = body.messages ?? [];
    const system = messages.find((m) => m.role === "system")?.content ?? "";
    const user = messages.find((m) => m.role === "user")?.content ?? "";
    const content = reply(system, user);
    return new Response(
      JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion",
        model: "test-model",
        choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }],
        usage: { prompt_tokens: 20, completion_tokens: 8, total_tokens: 28 },
      }),
      { headers: { "content-type": "application/json" } },
    );
  });
  globalThis.fetch = fake as unknown as typeof fetch;
  return fake;
}

/** A scripted improver backend: returns queued replies in order. */
export class ScriptedBackend implements LlmBackend {
  readonly name = "scripted";
  readonly prompts: ChatMessage[][] = [];
  private readonly replies: Array<string | ((messages: ChatMessage[]) => string)>;

  constructor(replies: Array<string | ((messages: ChatMessage[]) => string)>) {
    this.replies = [...replies];
  }

  requireModel(override: string | null): string {
    return override ?? "scripted";
  }

  async complete(messages: ChatMessage[]): Promise<string> {
    this.prompts.push(messages);
    const next = this.replies.shift();
    if (next === undefined) throw new Error("scripted backend ran out of replies");
    return typeof next === "function" ? next(messages) : next;
  }
}

/** An agent module that reads its system prompt from a source literal. */
export function agentModuleSource(systemPrompt: string): string {
  return (
    `const SYSTEM = ${JSON.stringify(systemPrompt)};\n` +
    "export async function run(input) {\n" +
    '  const res = await fetch("https://api.openai.com/v1/chat/completions", {\n' +
    '    method: "POST",\n' +
    "    body: JSON.stringify({\n" +
    '      model: "test-model",\n' +
    "      temperature: 0.2,\n" +
    "      messages: [\n" +
    '        { role: "system", content: SYSTEM },\n' +
    '        { role: "user", content: input ?? "" },\n' +
    "      ],\n" +
    "    }),\n" +
    "  });\n" +
    "  const data = await res.json();\n" +
    "  return data.choices[0].message.content;\n" +
    "}\n"
  );
}
