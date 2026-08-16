# Playbook

Lessons agentopt has learned about this agent. Loaded into the agent's
system prompt via agentopt.load_playbook(). Delete any line you disagree
with — it will not be re-added.

- When a prompt fails to follow a required output format, restructure it as an explicit step-by-step procedure with few-shot examples — including one that reproduces the failing input verbatim with its correct output. <!-- id:pb_2451f68c -->
- When a failing goal expects specific surface content in the reply (like a greeting by name), add an explicit prompt instruction with a concrete example of the required phrasing rather than relying on the model to infer it. <!-- id:pb_ce46c2ef -->
- When a reply must always contain a fixed phrase, instruct the prompt to append it unconditionally to every reply rather than relying on the model to infer when it's appropriate. <!-- id:pb_522ceee3 -->
