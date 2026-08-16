/** Shared model client (Fireworks via the OpenAI-compatible endpoint). */

const BASE_URL = process.env.FIREWORKS_BASE_URL ?? "https://api.fireworks.ai/inference/v1";
const MODEL = process.env.FIREWORKS_MODEL ?? "accounts/fireworks/models/kimi-k2p6";

export async function chat(systemPrompt, userText, temperature = 0.7) {
  const response = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${process.env.FIREWORKS_API_KEY ?? ""}`,
    },
    body: JSON.stringify({
      model: MODEL,
      temperature,
      max_tokens: 2000,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userText },
      ],
    }),
  });
  if (!response.ok) {
    throw new Error(`model call failed: HTTP ${response.status} ${await response.text()}`);
  }
  const data = await response.json();
  return data.choices?.[0]?.message?.content ?? "";
}
