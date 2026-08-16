/** Public API of the agentopt package. */

export { promptHash, contentHash } from "./hashing.js";
export { loadConfig, AgentoptConfigSchema, type AgentoptConfig } from "./models/config.js";
export { ManifestSchema, type Manifest, type ManifestEntry } from "./models/manifest.js";
export {
  parseTraceLine,
  dumpTraceLine,
  type TraceEvent,
  type LlmCall,
  type ToolCall,
  type RunMeta,
} from "./models/trace.js";
export { Workspace } from "./paths.js";
export { startWiretap, registerServer, type WiretapRun } from "./wiretap/session.js";
export { loadGoals, parseGoals, compileGoals, type Goals, type EvalSuite } from "./goals.js";
export { EvalEngine, renderScorecard, type Scorecard } from "./evals/engine.js";
export { NodeRunner, CommandRunner, runnerFor, type AgentRunner } from "./runner.js";
export { loadPlaybook, PlaybookStore } from "./playbook.js";
export { expose, instrument, type InstrumentSpec } from "./instrument.js";
export { listVersions, rollback, type VersionInfo } from "./writer/history.js";
export { main } from "./cli/index.js";
