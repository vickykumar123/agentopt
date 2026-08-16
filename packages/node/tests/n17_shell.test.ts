/** N17: the shell-words command tokenizer (parity with Python's shlex.split). */

import { describe, expect, it } from "vitest";
import { splitCommand } from "../src/shell.js";

describe("splitCommand (POSIX)", () => {
  const posix = (c: string) => splitCommand(c, "linux");

  it("splits on whitespace and consumes double quotes around spaced args", () => {
    expect(posix('python score.py "arg with spaces"')).toEqual([
      "python",
      "score.py",
      "arg with spaces",
    ]);
  });

  it("treats single quotes as literal (no expansion, keeps inner double quotes)", () => {
    expect(posix(`echo 'a "b" c'`)).toEqual(["echo", 'a "b" c']);
  });

  it("honors backslash escapes inside and outside double quotes", () => {
    expect(posix('run "a \\"quoted\\" word"')).toEqual(["run", 'a "quoted" word']);
    expect(posix("run a\\ b")).toEqual(["run", "a b"]); // escaped space stays one token
  });

  it("preserves an empty quoted token", () => {
    expect(posix('cmd "" x')).toEqual(["cmd", "", "x"]);
  });

  it("throws on an unterminated quote instead of silently mis-splitting", () => {
    expect(() => posix('cmd "oops')).toThrow(/unterminated/);
  });
});

describe("splitCommand (Windows)", () => {
  const win = (c: string) => splitCommand(c, "win32");

  it("keeps backslashes literal (path separators) and strips surrounding quotes", () => {
    expect(win('"C:\\Program Files\\node.exe" --flag')).toEqual([
      "C:\\Program Files\\node.exe",
      "--flag",
    ]);
  });

  it("groups spaced args by double quotes", () => {
    expect(win('python score.py "arg with spaces"')).toEqual([
      "python",
      "score.py",
      "arg with spaces",
    ]);
  });
});
