/** Entry point: input format "<user_id>|<message>". */

import { fetchUser } from "./src/db.mjs";
import { classify } from "./src/classify.mjs";
import { reply } from "./src/reply.mjs";

export async function run(inputText) {
  const raw = inputText || "1|hi there";
  const [userId, ...rest] = raw.split("|");
  const message = rest.join("|") || "hi there";
  const user = fetchUser(Number(userId || "1"));
  const intent = await classify(message);
  const answer = await reply(intent, message, user);
  return `[${intent}] ${answer}`;
}
