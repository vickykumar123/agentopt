/** AgentOpt adapter for the support application entry point. */

// Repository demo import. In an installed application use:
// import { expose } from "agentopt";
import { expose } from "../../dist/index.js";
import { run as runSupportAgent } from "./agent.mjs";

export const run = expose(runSupportAgent);
