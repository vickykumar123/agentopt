/**
 * The safe writer: one precise, reversible edit to the user's real file.
 *
 * Never string find-and-replace. The verified candidate's text is rendered
 * into a valid string/template-literal source, round-trip verified (the
 * rendered node must parse back to exactly the intended content), and
 * swapped in place of the manifest's node via ts-morph — every other byte
 * of the file stays identical. The file is snapshotted to history before
 * the write; any doubt aborts the apply rather than risking the user's code.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  Project,
  SyntaxKind,
  type Node,
  type ObjectLiteralExpression,
  type PropertyAssignment,
} from "ts-morph";
import { contentHash } from "../hashing.js";
import type { Candidate } from "../improver/candidates.js";
import { PromptLocator } from "../manifest/locator.js";
import { ManifestSchema, type ManifestEntry } from "../models/manifest.js";
import type { Workspace } from "../paths.js";
import { existsSync } from "node:fs";
import { recordPostHashes, snapshot } from "./history.js";

const SLOT = /\{([^{}]+)\}/g;

export class WriterError extends Error {}

export class SafeWriter {
  constructor(
    private readonly workspace: Workspace,
    private readonly protectedGlobs: string[] = [],
  ) {}

  /** Write the candidate to its file; returns the history version id. */
  apply(
    candidate: Candidate,
    entry: ManifestEntry,
    scores: { scoreBefore: number | null; scoreAfter: number | null },
  ): string {
    if (candidate.kind !== "prompt") {
      throw new WriterError(
        "params changes are recommendations (their source is not a file); only prompt changes are written",
      );
    }
    if (!entry.file || !entry.node_path || candidate.new_content == null) {
      throw new WriterError(`entry ${entry.id} is not file-writable`);
    }
    if (this.protectedGlobs.some((glob) => globMatch(glob, entry.file!))) {
      throw new WriterError(`${entry.file} matches a protected pattern — never edited`);
    }

    const path = join(this.workspace.projectRoot, entry.file);
    const originalSource = readFileSync(path, "utf-8");
    const slots = entry.slots ?? [];
    const isTemplate = entry.source_kind === "fstring";
    const rendered = renderNodeSource(candidate.new_content, slots, isTemplate);
    const newSource = replaceNode(originalSource, entry.file, entry.node_path, rendered);

    const version = snapshot(this.workspace, {
      description: candidate.description,
      entryId: entry.id,
      preImages: { [entry.file]: originalSource },
      scoreBefore: scores.scoreBefore,
      scoreAfter: scores.scoreAfter,
    });
    writeFileSync(path, newSource, "utf-8");
    recordPostHashes(this.workspace, version, [entry.file]);
    this.refreshManifestEntry(entry, candidate.new_content);
    return version;
  }

  /**
   * Apply a verified generation-param change to source deterministically, via
   * an AST edit of the model call — no code delegate, no git, works with any
   * backend. Returns the history version, or throws when the call can't be
   * located (the caller falls back to the delegate). Versioned and reversible
   * exactly like a prompt edit.
   */
  applyParams(
    paramOverrides: Record<string, unknown>,
    entry: ManifestEntry,
    promptText: string | null,
    callSiteLine: number | null,
    scores: { scoreBefore: number | null; scoreAfter: number | null },
    description: string,
  ): string {
    if (!entry.file) throw new WriterError(`entry ${entry.id} has no source file`);
    if (this.protectedGlobs.some((glob) => globMatch(glob, entry.file!))) {
      throw new WriterError(`${entry.file} matches a protected pattern — never edited`);
    }
    const path = join(this.workspace.projectRoot, entry.file);
    const originalSource = readFileSync(path, "utf-8");
    const newSource = setCallParams(
      originalSource,
      entry.file,
      promptText,
      callSiteLine,
      paramOverrides,
    );
    if (newSource === null) {
      throw new WriterError(
        `could not locate the model call for ${entry.id} to set params in source`,
      );
    }
    const version = snapshot(this.workspace, {
      description,
      entryId: entry.id,
      preImages: { [entry.file]: originalSource },
      scoreBefore: scores.scoreBefore,
      scoreAfter: scores.scoreAfter,
    });
    writeFileSync(path, newSource, "utf-8");
    recordPostHashes(this.workspace, version, [entry.file]);
    return version;
  }

  /** Point the manifest entry at the rewritten node so drift detection keeps
   * working without a full re-observe. */
  private refreshManifestEntry(entry: ManifestEntry, newContent: string): void {
    const manifestPath = this.workspace.manifestFile;
    if (!existsSync(manifestPath) || !entry.file) return;
    const locator = new PromptLocator(this.workspace.projectRoot);
    const located = locator
      .candidatesIn(entry.file)
      .find((c) => c.template === newContent || c.staticValue === newContent);
    if (!located) return;
    const manifest = ManifestSchema.parse(JSON.parse(readFileSync(manifestPath, "utf-8")));
    for (const existing of manifest.entries) {
      if (existing.id === entry.id) {
        existing.line = located.line;
        existing.node_path = `${located.nodeType}@${located.line}:${located.column}`;
        existing.content_hash = contentHash(located.source);
      }
    }
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), "utf-8");
  }
}

/**
 * A valid TS/JS source snippet for the candidate content: a double-quoted
 * string for literals, a backtick template with `${slot}` interpolations for
 * templates. Round-trip verified before a single byte is written.
 */
export function renderNodeSource(
  content: string,
  slots: string[],
  isTemplate: boolean,
): string {
  if (!isTemplate) {
    const rendered = JSON.stringify(content);
    if ((JSON.parse(rendered) as string) !== content) {
      throw new WriterError("rendered literal failed round-trip verification");
    }
    return rendered;
  }

  const found = [...content.matchAll(SLOT)].map((m) => m[1]!);
  const stray = found.filter((name) => !slots.includes(name));
  if (stray.length || (content.match(/\{/g) ?? []).length !== found.length) {
    throw new WriterError("candidate contains braces that are not known slots");
  }

  const parts = content.split(SLOT);
  let rendered = "`";
  for (let index = 0; index < parts.length; index++) {
    if (index % 2 === 1) {
      rendered += `\${${parts[index]!}}`;
    } else {
      rendered += parts[index]!
        .replaceAll("\\", "\\\\")
        .replaceAll("`", "\\`")
        .replaceAll("${", "\\${");
    }
  }
  rendered += "`";

  if (templateOf(rendered) !== content) {
    throw new WriterError("rendered template failed round-trip verification");
  }
  return rendered;
}

/** Re-derive the display template of a rendered template-literal source. */
function templateOf(renderedSource: string): string {
  const project = new Project({ useInMemoryFileSystem: true });
  const source = project.createSourceFile("check.ts", `const value = ${renderedSource};`);
  const template = source.getFirstDescendantByKind(SyntaxKind.TemplateExpression);
  if (template) {
    const parts = [template.getHead().getLiteralText()];
    for (const span of template.getTemplateSpans()) {
      parts.push(`{${span.getExpression().getText().trim()}}`, span.getLiteral().getLiteralText());
    }
    return parts.join("");
  }
  const plain = source.getFirstDescendantByKind(SyntaxKind.NoSubstitutionTemplateLiteral);
  if (plain) return plain.getLiteralText();
  throw new WriterError("rendered template failed to parse");
}

/** Replace exactly the node the manifest points at; everything else stays. */
export function replaceNode(
  source: string,
  relativeFile: string,
  nodePath: string,
  newNodeSource: string,
): string {
  const match = /^(\w+)@(\d+):(\d+)$/.exec(nodePath);
  if (!match) throw new WriterError(`malformed node path: ${JSON.stringify(nodePath)}`);
  const [, nodeType, lineText, columnText] = match;
  const line = Number(lineText);
  const column = Number(columnText);

  const project = new Project({ useInMemoryFileSystem: true, compilerOptions: { allowJs: true } });
  const sourceFile = project.createSourceFile(relativeFile.split("/").pop() ?? "file.ts", source);

  const kinds =
    nodeType === "StringLiteral"
      ? [SyntaxKind.StringLiteral, SyntaxKind.NoSubstitutionTemplateLiteral]
      : nodeType === "Concatenation"
        ? [SyntaxKind.BinaryExpression]
        : [SyntaxKind.TemplateExpression];
  // A concatenation and its first operand start at the same position; pick
  // the widest node of the target kind so the whole `a + b + c` is replaced.
  let target: Node | undefined;
  let widest = -1;
  for (const kind of kinds) {
    for (const node of sourceFile.getDescendantsOfKind(kind)) {
      const start = sourceFile.getLineAndColumnAtPos(node.getStart());
      if (start.line !== line || start.column !== column) continue;
      const width = node.getEnd() - node.getStart();
      if (width > widest) {
        widest = width;
        target = node;
      }
    }
  }
  if (!target) {
    throw new WriterError(
      `node ${nodePath} not found — the file changed since the manifest was built; ` +
        "run 'agentopt observe' to refresh it",
    );
  }
  target.replaceWithText(newNodeSource);
  return sourceFile.getFullText();
}

/**
 * Set generation parameters on the model call that sends `promptSnippet`.
 *
 * Deterministic AST edit — no LLM, no git. Finds the options object of the
 * `create({ model, messages, ... })` call carrying that prompt (or, failing
 * that, the call nearest `callSiteLine`) and sets/replaces each param on it.
 * Returns the rewritten source, or null when the call can't be located (the
 * caller then falls back to the code delegate).
 */
export function setCallParams(
  source: string,
  relativeFile: string,
  promptSnippet: string | null,
  callSiteLine: number | null,
  paramOverrides: Record<string, unknown>,
): string | null {
  const project = new Project({ useInMemoryFileSystem: true, compilerOptions: { allowJs: true } });
  const sourceFile = project.createSourceFile(relativeFile.split("/").pop() ?? "file.ts", source);

  const options = locateOptionsObject(sourceFile, promptSnippet, callSiteLine);
  if (!options) return null;

  let changed = false;
  for (const [key, value] of Object.entries(paramOverrides)) {
    const literal = typeof value === "string" ? JSON.stringify(value) : String(value);
    const existing = options.getProperty(key);
    if (existing && existing.getKind() === SyntaxKind.PropertyAssignment) {
      const assignment = existing as PropertyAssignment;
      if (assignment.getInitializer()?.getText() === literal) continue;
      assignment.setInitializer(literal);
    } else {
      options.addPropertyAssignment({ name: key, initializer: literal });
    }
    changed = true;
  }
  return changed ? sourceFile.getFullText() : null;
}

/** The `{ model, messages, ... }` options object of the target model call. */
function locateOptionsObject(
  sourceFile: import("ts-morph").SourceFile,
  promptSnippet: string | null,
  callSiteLine: number | null,
): ObjectLiteralExpression | null {
  // Primary anchor: walk up from the prompt literal to the nearest object that
  // is a direct argument of a call — that is the model call's options object.
  // Key off the first line only: a multi-line key breaks on CRLF-vs-LF, since
  // the parsed prompt uses \n while the raw source may use \r\n.
  const key = promptSnippet?.split(/\r?\n/)[0]?.slice(0, 60);
  if (key) {
    const literalKinds = [
      SyntaxKind.StringLiteral,
      SyntaxKind.NoSubstitutionTemplateLiteral,
      SyntaxKind.TemplateExpression,
    ];
    for (const kind of literalKinds) {
      for (const literal of sourceFile.getDescendantsOfKind(kind)) {
        if (!literal.getText().includes(key)) continue;
        let node: Node | undefined = literal.getParent();
        while (node) {
          if (
            node.getKind() === SyntaxKind.ObjectLiteralExpression &&
            node.getParent()?.getKind() === SyntaxKind.CallExpression
          ) {
            return node as ObjectLiteralExpression;
          }
          node = node.getParent();
        }
      }
    }
  }
  // Fallback: the call nearest the traced call-site line; take its object arg.
  if (callSiteLine != null) {
    for (const call of sourceFile.getDescendantsOfKind(SyntaxKind.CallExpression)) {
      const line = sourceFile.getLineAndColumnAtPos(call.getStart()).line;
      if (Math.abs(line - callSiteLine) > 2) continue;
      const objectArg = call
        .getArguments()
        .reverse()
        .find((arg) => arg.getKind() === SyntaxKind.ObjectLiteralExpression);
      if (objectArg) return objectArg as ObjectLiteralExpression;
    }
  }
  return null;
}

function globMatch(glob: string, file: string): boolean {
  const pattern = glob
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replaceAll("**", " ")
    .replaceAll("*", "[^/]*")
    .replaceAll(" ", ".*");
  return new RegExp(`^${pattern}$`).test(file);
}
