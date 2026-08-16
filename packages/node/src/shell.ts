/**
 * Split a command line into argv the way a shell would — the Node counterpart of
 * Python's `shlex.split`, so both packages tokenize `run.command`,
 * `run.http.start`, and `judge_command` identically.
 *
 * POSIX rules off-Windows: single quotes are literal, double quotes allow
 * backslash escapes, and a bare backslash escapes the next character. On Windows
 * backslashes are literal path separators (`C:\Program Files\node.exe`), so we
 * only group tokens by quotes and strip surrounding double quotes — matching
 * CPython's `shlex.split(cmd, posix=False)` followed by `.strip('"')`.
 *
 * `platform` defaults to the host but is injectable so both branches are testable
 * on any OS.
 */
export function splitCommand(command: string, platform: string = process.platform): string[] {
  return platform === "win32" ? splitWindows(command) : splitPosix(command);
}

function splitPosix(command: string): string[] {
  const argv: string[] = [];
  let token = "";
  let started = false; // distinguishes an empty quoted token ("") from no token
  let i = 0;
  while (i < command.length) {
    const c = command[i]!;
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      if (started) {
        argv.push(token);
        token = "";
        started = false;
      }
      i++;
    } else if (c === "'") {
      started = true;
      i++;
      while (i < command.length && command[i] !== "'") token += command[i++];
      if (i >= command.length) throw new Error(`unterminated single quote: ${command}`);
      i++; // closing '
    } else if (c === '"') {
      started = true;
      i++;
      while (i < command.length && command[i] !== '"') {
        if (command[i] === "\\" && i + 1 < command.length && '"\\$`'.includes(command[i + 1]!)) {
          token += command[i + 1];
          i += 2;
        } else {
          token += command[i++];
        }
      }
      if (i >= command.length) throw new Error(`unterminated double quote: ${command}`);
      i++; // closing "
    } else if (c === "\\") {
      started = true;
      token += i + 1 < command.length ? command[++i] : "\\";
      i++;
    } else {
      started = true;
      token += c;
      i++;
    }
  }
  if (started) argv.push(token);
  return argv;
}

function splitWindows(command: string): string[] {
  const argv: string[] = [];
  let token = "";
  let started = false;
  let quote = ""; // "'" or '"' while inside a quoted span (quotes are retained)
  for (const c of command) {
    if (quote) {
      token += c;
      if (c === quote) quote = "";
    } else if (c === "'" || c === '"') {
      started = true;
      token += c;
      quote = c;
    } else if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      if (started) {
        argv.push(token);
        token = "";
        started = false;
      }
    } else {
      started = true;
      token += c;
    }
  }
  if (started) argv.push(token);
  // Strip surrounding double quotes, matching Python's `.strip('"')`.
  return argv.map((t) => t.replace(/^"+/, "").replace(/"+$/, ""));
}
