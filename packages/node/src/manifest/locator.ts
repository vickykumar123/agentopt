/**
 * Static location of prompt text in TypeScript/JavaScript source, via
 * ts-morph. Runtime strings from traces are matched against string nodes:
 * plain literals match exactly; template literals match with their
 * `${expressions}` treated as wildcard slots. ts-morph preserves exact
 * formatting, so a located node can later be rewritten without touching
 * anything around it.
 */

import { readdirSync, statSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";
import { Node, Project, SyntaxKind, type SourceFile } from "ts-morph";
import { matchTemplatePattern } from "../pattern.js";

/** Below this much static text (e.g. `${x}`) almost anything would match. */
export const MIN_STATIC_MATCH_LENGTH = 8;

const SCAN_EXCLUDED_DIRS = new Set([
  "node_modules",
  ".agentopt",
  ".git",
  "dist",
  "build",
  ".next",
]);
const SOURCE_EXTENSIONS = new Set([".ts", ".tsx", ".js", ".mjs", ".cjs"]);

export interface StringCandidate {
  file: string;
  line: number;
  column: number;
  nodeType: "StringLiteral" | "TemplateLiteral" | "Concatenation";
  source: string;
  staticValue: string | null;
  pattern: string | null;
  slots: string[];
  /** Human/display template with `{slot}` placeholders. */
  template: string;
}

export function nodePathOf(candidate: StringCandidate): string {
  return `${candidate.nodeType}@${candidate.line}:${candidate.column}`;
}

export function candidateMatches(
  candidate: StringCandidate,
  content: string,
): boolean {
  if (candidate.staticValue !== null) return candidate.staticValue === content;
  if (candidate.pattern !== null) {
    return matchTemplatePattern(candidate.pattern, content) !== null;
  }
  return false;
}

export class PromptLocator {
  private readonly projectRoot: string;
  private readonly excludeGlobs: RegExp[];
  private readonly project = new Project({
    useInMemoryFileSystem: false,
    compilerOptions: { allowJs: true },
  });
  private readonly index = new Map<string, StringCandidate[]>();

  constructor(projectRoot: string, excludeGlobs: string[] = []) {
    this.projectRoot = resolve(projectRoot);
    this.excludeGlobs = excludeGlobs.map(globToRegExp);
  }

  private isExcluded(relativeFile: string): boolean {
    return this.excludeGlobs.some((glob) => glob.test(relativeFile));
  }

  candidatesIn(relativeFile: string): StringCandidate[] {
    if (this.isExcluded(relativeFile)) return [];
    const cached = this.index.get(relativeFile);
    if (cached) return cached;
    const candidates = this.indexFile(relativeFile);
    this.index.set(relativeFile, candidates);
    return candidates;
  }

  /**
   * All candidates matching `content`. If any match lives in
   * `preferredFile` (the call site), matches narrow to that file.
   */
  find(content: string, preferredFile: string | null): StringCandidate[] {
    if (preferredFile !== null) {
      const local = this.candidatesIn(preferredFile).filter((candidate) =>
        candidateMatches(candidate, content),
      );
      if (local.length > 0) return local;
    }
    const matches: StringCandidate[] = [];
    for (const file of this.projectFiles()) {
      for (const candidate of this.candidatesIn(file)) {
        if (candidateMatches(candidate, content)) matches.push(candidate);
      }
    }
    return matches;
  }

  /** The candidate a manifest entry points at, re-located by node path. */
  candidateAt(relativeFile: string, nodePath: string): StringCandidate | null {
    return (
      this.candidatesIn(relativeFile).find(
        (candidate) => nodePathOf(candidate) === nodePath,
      ) ?? null
    );
  }

  private indexFile(relativeFile: string): StringCandidate[] {
    const absolute = join(this.projectRoot, relativeFile);
    let source: SourceFile;
    try {
      source =
        this.project.getSourceFile(absolute) ??
        this.project.addSourceFileAtPath(absolute);
    } catch {
      return [];
    }
    const candidates: StringCandidate[] = [];

    const pushStatic = (node: Node, value: string): void => {
      // A fragment of a larger `a + b` prompt is not independently editable;
      // the whole concatenation is indexed as one candidate below.
      if (!value || isPartOfLargerConcat(node)) return;
      const start = source.getLineAndColumnAtPos(node.getStart());
      candidates.push({
        file: relativeFile,
        line: start.line,
        column: start.column,
        nodeType: "StringLiteral",
        source: node.getText(),
        staticValue: value,
        pattern: null,
        slots: [],
        template: value,
      });
    };

    for (const literal of source.getDescendantsOfKind(SyntaxKind.StringLiteral)) {
      pushStatic(literal, literal.getLiteralValue());
    }
    for (const literal of source.getDescendantsOfKind(
      SyntaxKind.NoSubstitutionTemplateLiteral,
    )) {
      pushStatic(literal, literal.getLiteralValue());
    }

    for (const template of source.getDescendantsOfKind(
      SyntaxKind.TemplateExpression,
    )) {
      if (isPartOfLargerConcat(template)) continue; // folded into the concat below
      const candidate = partsToCandidate(
        leafToParts(template),
        template,
        source,
        relativeFile,
        "TemplateLiteral",
      );
      if (candidate) candidates.push(candidate);
    }

    // String concatenations (`"a" + name + "b"`) are constant-folded: static
    // literals become fixed text, non-literal operands become slots — the
    // same representation as a template literal, so they are both matchable
    // at runtime and rewritable in place.
    for (const bin of source.getDescendantsOfKind(SyntaxKind.BinaryExpression)) {
      if (bin.getOperatorToken().getKind() !== SyntaxKind.PlusToken) continue;
      if (isPartOfLargerConcat(bin)) continue; // only the outermost concat
      const leaves = flattenPlus(bin);
      if (!leaves.some(isStringLiteralLike)) continue; // not string concatenation
      const candidate = partsToCandidate(
        leaves.flatMap(leafToParts),
        bin,
        source,
        relativeFile,
        "Concatenation",
      );
      if (candidate) candidates.push(candidate);
    }

    return candidates;
  }

  projectFiles(): string[] {
    const files: string[] = [];
    const walk = (dir: string): void => {
      let entries: string[];
      try {
        entries = readdirSync(dir);
      } catch {
        return;
      }
      for (const entry of entries.sort()) {
        const full = join(dir, entry);
        let stats;
        try {
          stats = statSync(full);
        } catch {
          continue;
        }
        if (stats.isDirectory()) {
          if (!SCAN_EXCLUDED_DIRS.has(entry)) walk(full);
        } else if (SOURCE_EXTENSIONS.has(extensionOf(entry))) {
          const rel = relative(this.projectRoot, full).split(sep).join("/");
          if (!this.isExcluded(rel)) files.push(rel);
        }
      }
    };
    walk(this.projectRoot);
    return files;
  }
}

function extensionOf(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot === -1 ? "" : name.slice(dot);
}

/** One piece of a folded prompt: fixed text, or a runtime-filled slot. */
type PromptPart = { kind: "static"; text: string } | { kind: "slot"; expr: string };

/**
 * Assemble a candidate from ordered parts. All-static parts yield an exact
 * literal; any slot yields a `(.*?)` pattern plus a `{slot}` display
 * template — the shared shape for template literals and concatenations.
 */
function partsToCandidate(
  parts: PromptPart[],
  node: Node,
  source: SourceFile,
  relativeFile: string,
  nodeType: StringCandidate["nodeType"],
): StringCandidate | null {
  const patternParts: string[] = [];
  const templateParts: string[] = [];
  const slots: string[] = [];
  let staticLength = 0;
  for (const part of parts) {
    if (part.kind === "static") {
      patternParts.push(escapeRegExp(part.text));
      templateParts.push(part.text);
      staticLength += part.text.length;
    } else {
      slots.push(part.expr);
      patternParts.push("(.*?)");
      templateParts.push(`{${part.expr}}`);
    }
  }
  const start = source.getLineAndColumnAtPos(node.getStart());
  const base = {
    file: relativeFile,
    line: start.line,
    column: start.column,
    nodeType,
    source: node.getText(),
  };
  if (slots.length === 0) {
    const value = templateParts.join("");
    return value ? { ...base, staticValue: value, pattern: null, slots: [], template: value } : null;
  }
  // Below the static floor almost anything matches — not a reliable anchor.
  if (staticLength < MIN_STATIC_MATCH_LENGTH) return null;
  return {
    ...base,
    staticValue: null,
    pattern: patternParts.join(""),
    slots,
    template: templateParts.join(""),
  };
}

/** Flatten a left-associative `+` chain (through parentheses) into operands. */
function flattenPlus(node: Node): Node[] {
  if (
    Node.isBinaryExpression(node) &&
    node.getOperatorToken().getKind() === SyntaxKind.PlusToken
  ) {
    return [...flattenPlus(node.getLeft()), ...flattenPlus(node.getRight())];
  }
  if (Node.isParenthesizedExpression(node)) return flattenPlus(node.getExpression());
  return [node];
}

/** One concatenation operand as parts: literals fold to text, else a slot. */
function leafToParts(leaf: Node): PromptPart[] {
  if (Node.isStringLiteral(leaf) || Node.isNoSubstitutionTemplateLiteral(leaf)) {
    return [{ kind: "static", text: leaf.getLiteralValue() }];
  }
  if (Node.isTemplateExpression(leaf)) {
    const parts: PromptPart[] = [{ kind: "static", text: leaf.getHead().getLiteralText() }];
    for (const span of leaf.getTemplateSpans()) {
      parts.push({ kind: "slot", expr: span.getExpression().getText().trim() });
      parts.push({ kind: "static", text: span.getLiteral().getLiteralText() });
    }
    return parts;
  }
  // A numeric operand coerces to its source text; anything else is dynamic.
  if (Node.isNumericLiteral(leaf)) return [{ kind: "static", text: leaf.getText() }];
  return [{ kind: "slot", expr: leaf.getText().trim() }];
}

function isStringLiteralLike(node: Node): boolean {
  return (
    Node.isStringLiteral(node) ||
    Node.isNoSubstitutionTemplateLiteral(node) ||
    Node.isTemplateExpression(node)
  );
}

/** True when `node` is itself an operand of a surrounding `+` concatenation. */
function isPartOfLargerConcat(node: Node): boolean {
  let parent = node.getParent();
  while (parent && Node.isParenthesizedExpression(parent)) parent = parent.getParent();
  return (
    parent !== undefined &&
    Node.isBinaryExpression(parent) &&
    parent.getOperatorToken().getKind() === SyntaxKind.PlusToken
  );
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function globToRegExp(glob: string): RegExp {
  const escaped = glob
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replaceAll("**", " ")
    .replaceAll("*", "[^/]*")
    .replaceAll(" ", ".*");
  return new RegExp(`^${escaped}$`);
}
