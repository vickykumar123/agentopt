/**
 * Hermetic wiretap demo — no API key needed. A local mock model endpoint
 * stands in for the provider; the wiretap captures the agent's calls exactly
 * as it would against the real API. Run with:
 *
 *   node examples/wiretap-demo.mjs
 */

import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { startWiretap } from "../dist/wiretap/session.js";

// 1. a stand-in model endpoint (OpenAI response shape)
const server = createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => (body += chunk));
  req.on("end", () => {
    const { messages = [] } = JSON.parse(body || "{}");
    const user = messages.find((m) => m.role === "user")?.content ?? "";
    res.setHeader("content-type", "application/json");
    res.end(
      JSON.stringify({
        model: "mock-model",
        choices: [
          {
            message: { role: "assistant", content: `intent: ${/refund|money back/i.test(user) ? "refund" : "other"}` },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 21, completion_tokens: 4 },
      }),
    );
  });
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;

// 2. a tiny agent that calls the model through plain fetch
async function agent(input) {
  const response = await fetch(`http://127.0.0.1:${port}/v1/chat/completions`, {
    method: "POST",
    body: JSON.stringify({
      model: "mock-model",
      temperature: 0.3,
      messages: [
        { role: "system", content: "You classify fan messages into intents." },
        { role: "user", content: input },
      ],
    }),
  });
  return (await response.json()).choices[0].message.content;
}

// 3. run it under capture
const run = startWiretap({ projectRoot: process.cwd(), inputText: "money back pls??" });
const output = await agent("money back pls??");
await run.stop();
server.close();

console.log(`agent output: ${output}\n`);
console.log(`trace: ${run.tracePath}\n`);
for (const line of readFileSync(run.tracePath, "utf-8").trim().split("\n")) {
  const event = JSON.parse(line);
  if (event.kind === "run_meta") console.log(`[run_meta] mode=${event.mode} input=${JSON.stringify(event.input)}`);
  if (event.kind === "llm_call") {
    console.log(`[llm_call] ${event.provider}/${event.model}  ${event.latency_ms?.toFixed(0)}ms`);
    console.log(`  system: ${event.messages[0].content}`);
    console.log(`  reply:  ${event.response?.content}`);
    console.log(`  usage:  ${event.usage?.input_tokens} in / ${event.usage?.output_tokens} out`);
    console.log(`  called from: ${event.call_site?.file}:${event.call_site?.line}`);
  }
}
