/** Step 1: classify the customer's message into an intent. */

import { chat } from "./llm.mjs";

const CLASSIFY_PROMPT = "Classify the customer message into exactly one of these categories: billing, greeting, refund, shipping.\n\nEach input line has the form <id>|<text>. The leading <id>| is a record identifier — ignore it and classify only the text that follows.\n\nExamples:\n2|where is my package -> shipping\n7|hi there -> greeting\n3|i want my money back -> refund\n5|my invoice is wrong -> billing\n\nReply with ONLY the full category name (billing, greeting, refund, or shipping), a single lowercase word, nothing else.";

const KNOWN = ["refund", "shipping", "billing", "greeting", "other"];

export async function classify(message) {
  const label = (await chat(CLASSIFY_PROMPT, message, 0.2)).trim().toLowerCase();
  return KNOWN.includes(label) ? label : "other";
}
