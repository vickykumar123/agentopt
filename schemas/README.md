# Shared Schemas

Language-neutral contracts both packages (pip + npm) must obey. The Python and
TypeScript implementations are independent, but the files they read and write
are byte-compatible, defined here.

| File | Contract for | Status |
|---|---|---|
| `trace.schema.json` | Trace store entries (`.agentopt/traces/*.jsonl`) | ✅ defined |
| `manifest.schema.json` | Prompt/param/tool → file:line map (`.agentopt/manifest.json`) | ✅ defined |
| `config.schema.json` | `agentopt.yaml` | ✅ defined |
| `ledger.schema.json` | Reward memory (`.agentopt/ledger.json`) | ✅ defined |
| `approved-case.schema.json` | Durable cases (`agentopt.cases.yaml`) | ✅ defined |
| `decision.schema.json` | Candidate gate decisions | ✅ defined |
| `version-report.schema.json` | Machine-readable version report contract | ✅ defined |
| `goals-format.md` | GOALS.md format specification (sections, semantics) | ✅ defined |
| `hashing.md` | Canonical prompt/content fingerprint recipes | ✅ defined |
| `scorecard.schema.json` | Eval reports | ✅ implemented |
| `history.md` | Version snapshot layout (`.agentopt/history/`) | ✅ implemented |
| `run-evidence.schema.json` | Normalized execution and case-grade evidence | contract defined; standalone artifact not emitted yet |

A cross-language test suite asserts both implementations produce compatible
artifacts.
