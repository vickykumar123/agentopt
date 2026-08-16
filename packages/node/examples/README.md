# Node examples

- [multi-agent-demo](multi-agent-demo/README.md) — recommended first example;
  dependency-free evaluation of a coordinator and four named agents using
  `expose({...})`, `@surface`, structured trajectories, and critical holdouts.
- [`wiretap-demo.mjs`](wiretap-demo.mjs) — see the wiretap capture an agent's
  model calls, no API key required (`npm run build` first, then
  `node examples/wiretap-demo.mjs`).
- [support-demo](support-demo/README.md) — the full loop on a realistic
  multi-file agent with SQLite: observe → eval → improve → history → rollback.
  It needs a model key (or OpenAI-compatible endpoint) and an improver backend.

Provider-specific Python examples and a guide for choosing a starting point are
in the repository-level [examples index](../../../examples/README.md).
