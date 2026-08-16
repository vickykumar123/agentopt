/** Small deterministic multi-agent application; no model key is required. */

export function router(input) {
  const message = (input ?? "").toLowerCase();
  let destination = "account";
  if (["refund", "price", "plan", "invoice"].some((word) => message.includes(word))) {
    destination = "billing";
  } else if (["password", "login", "log in", "error"].some((word) => message.includes(word))) {
    destination = "technical";
  }
  return { output: destination, tools: [], route: "direct" };
}

export function billing(input) {
  const message = (input ?? "").toLowerCase();
  if (message.includes("refund")) {
    return {
      output: "Refunds are covered by our 30-day money-back policy.",
      tools: ["lookup_order"],
      route: "tools",
    };
  }
  return {
    output: "The Pro plan costs $19.99 per month.",
    tools: [],
    route: "direct",
  };
}

export function technical(_input) {
  return {
    output: "I can send a secure password reset link.",
    tools: [],
    route: "direct",
  };
}

export function account(_input) {
  return {
    output: "I will connect you with a human account specialist.",
    tools: ["escalate_to_human"],
    route: "tools",
  };
}

export function coordinator(input) {
  const destination = router(input).output;
  const handlers = { billing, technical, account };
  return handlers[destination](input);
}
