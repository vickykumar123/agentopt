/** Expose the complete workflow and each specialist as named surfaces. */

// Repository demo import. In an installed application use:
// import { expose } from "agentopt";
import { expose } from "../../dist/index.js";
import { account, billing, coordinator, router, technical } from "./agents.mjs";

// The map form lets AgentOpt dispatch @router cases directly and validate that
// every surface named in GOALS.md has a real handler.
export const run = expose({
  default: coordinator,
  router,
  billing,
  technical,
  account,
});
