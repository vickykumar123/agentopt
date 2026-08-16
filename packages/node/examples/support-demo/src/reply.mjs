/** Step 2: write the answer, grounded in the customer's account data. */

import { chat } from "./llm.mjs";

export async function reply(intent, message, user) {
  const prompt = `You are a support agent for a fan-subscription platform.
The customer is ${user.name} on the ${user.plan} plan at ${user.price}/month.
The message was classified as: ${intent}.
Answer the customer's question briefly.`;
  return chat(prompt, message, 0.7);
}
