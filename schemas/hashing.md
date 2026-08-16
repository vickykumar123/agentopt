# Canonical Hashing Specification

Hashes link runtime observations (traces) to static code locations (manifest)
and detect drift between runs. Both language implementations must produce
identical hashes for identical content.

## `prompt_hash` (trace events, manifest `prompt_hashes`)

Identifies a prompt as sent at runtime.

1. Take the `messages` array of the LLM call and canonicalize each message's
   content to a string:
   - string content → unchanged;
   - a list of parts → each text part contributes its text, every non-text
     part (image, embedded tool result) contributes the fixed placeholder
     `[non-text]`; parts are joined with `"\n"`;
   - any other content value → the placeholder `[non-text]`.
   Provider-specific top-level system prompts (e.g. Anthropic's `system`
   parameter) are canonicalized as a leading message with role `system`.
2. For each message, form the string `role + "\n" + content`.
3. Join all message strings with `"\n---\n"`.
4. Hash: SHA-256 of the UTF-8 encoding; lowercase hex digest.

Trace files store the canonicalized messages, so a stored trace always hashes
to its own `prompt_hash`.

## `content_hash` (manifest entries)

Identifies the static source of a tunable value for drift detection.

1. Take the exact source text of the AST node (for `params`: the JSON
   serialization of the normalized param object with sorted keys).
2. Hash: SHA-256 of the UTF-8 encoding; lowercase hex digest.

No whitespace normalization is applied: any byte change in the node is a
change, and the manifest re-baselines rather than guessing intent.
