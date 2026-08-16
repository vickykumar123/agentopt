"""The safe writer: one precise, reversible edit to the user's real file.

Never string find-and-replace. The verified candidate's text is rendered into
a valid Python string node, round-trip verified (the rendered node must parse
back to exactly the intended content), and swapped in place of the manifest's
node via libcst — every other byte of the file stays identical. The file is
snapshotted to history before the write and syntax-checked after; any doubt
aborts the apply rather than risking the user's code.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from agentopt.hashing import content_hash
from agentopt.manifest.locator import PromptLocator, StringCandidate
from agentopt.models.manifest import Manifest, ManifestEntry
from agentopt.paths import Workspace
from agentopt.writer.history import History

if TYPE_CHECKING:
    from agentopt.improver.candidates import Candidate

_SLOT = re.compile(r"\{([^{}]+)\}")


class WriterError(Exception):
    """The change could not be applied safely; nothing was written."""


class SafeWriter:
    """Applies verified candidates to files, snapshot-first."""

    def __init__(self, workspace: Workspace, protected: tuple[str, ...] = ()) -> None:
        self._workspace = workspace
        self._project_root = workspace.project_root
        self._protected = protected
        self._history = History(workspace)

    def apply(
        self,
        candidate: Candidate,
        entry: ManifestEntry,
        *,
        score_before: float | None,
        score_after: float | None,
    ) -> str:
        """Write the candidate to its file; returns the history version id."""
        if candidate.kind != "prompt":
            raise WriterError(
                "params changes are recommendations (their source is not a file); "
                "only prompt changes are written"
            )
        if entry.file is None or entry.node_path is None or candidate.new_content is None:
            raise WriterError(f"entry {entry.id} is not file-writable")
        import fnmatch

        if any(fnmatch.fnmatch(entry.file, glob) for glob in self._protected):
            raise WriterError(f"{entry.file} matches a protected pattern — never edited")

        path = self._project_root / entry.file
        original_source = path.read_text(encoding="utf-8")
        new_node_source = _render_node_source(
            candidate.new_content, entry.slots, is_fstring=entry.source_kind == "fstring"
        )
        new_source = _replace_node(original_source, entry.node_path, new_node_source)

        try:
            cst.parse_module(new_source)
        except cst.ParserSyntaxError as exc:
            raise WriterError(f"edit would break {entry.file}: {exc}") from exc

        version = self._history.snapshot(
            description=candidate.description,
            entry_id=entry.id,
            pre_images={entry.file: original_source},
            score_before=score_before,
            score_after=score_after,
        )
        path.write_text(new_source, encoding="utf-8")
        self._history.record_post_hashes(version, self._project_root, [entry.file])
        self._refresh_manifest_entry(entry, candidate.new_content)
        return version

    def apply_params(
        self,
        param_overrides: dict[str, object],
        entry: ManifestEntry,
        prompt_text: str | None,
        call_site_line: int | None,
        *,
        score_before: float | None,
        score_after: float | None,
        description: str,
    ) -> str:
        """Apply a verified generation-param change to source deterministically
        via an AST edit of the model call — no code delegate, no git, any
        backend. Returns the history version, or raises when the call can't be
        located (the caller falls back to the delegate)."""
        if entry.file is None:
            raise WriterError(f"entry {entry.id} has no source file")
        import fnmatch

        if any(fnmatch.fnmatch(entry.file, glob) for glob in self._protected):
            raise WriterError(f"{entry.file} matches a protected pattern — never edited")

        path = self._project_root / entry.file
        original_source = path.read_text(encoding="utf-8")
        new_source = set_call_params(
            original_source, prompt_text, call_site_line, param_overrides
        )
        if new_source is None:
            raise WriterError(
                f"could not locate the model call for {entry.id} to set params in source"
            )
        try:
            cst.parse_module(new_source)
        except cst.ParserSyntaxError as exc:
            raise WriterError(f"edit would break {entry.file}: {exc}") from exc

        version = self._history.snapshot(
            description=description,
            entry_id=entry.id,
            pre_images={entry.file: original_source},
            score_before=score_before,
            score_after=score_after,
        )
        path.write_text(new_source, encoding="utf-8")
        self._history.record_post_hashes(version, self._project_root, [entry.file])
        return version

    def _refresh_manifest_entry(self, entry: ManifestEntry, new_content: str) -> None:
        """Point the manifest entry at the rewritten node so drift detection
        keeps working without a full re-observe."""
        manifest_path = self._workspace.manifest_file
        if not manifest_path.is_file():
            return
        located = self._locate_new_node(entry, new_content)
        if located is None:
            return
        manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        for existing in manifest.entries:
            if existing.id == entry.id:
                existing.line = located.line
                existing.node_path = located.node_path
                existing.content_hash = content_hash(located.source)
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def _locate_new_node(self, entry: ManifestEntry, new_content: str) -> StringCandidate | None:
        if entry.file is None:
            return None
        locator = PromptLocator(self._project_root)
        for located in locator.candidates_in(entry.file):
            if located.template == new_content or located.static_value == new_content:
                return located
        return None


def _render_node_source(content: str, slots: list[str], *, is_fstring: bool) -> str:
    """A valid Python string-literal source for the candidate content.

    Round-trip verified: the rendered source is parsed and its value/template
    re-derived; any mismatch aborts before a single byte is written.
    """
    if not is_fstring:
        rendered = repr(content)
        parsed = cst.parse_expression(rendered)
        if not isinstance(parsed, cst.SimpleString) or parsed.evaluated_value != content:
            raise WriterError("rendered literal failed round-trip verification")
        return rendered

    stray = [b for b in _SLOT.findall(content) if b not in slots]
    if stray or content.count("{") != len(_SLOT.findall(content)):
        raise WriterError("candidate contains braces that are not known slots")

    quote = '"""' if ("\n" in content or '"' in content) else '"'
    body = content.replace("\\", "\\\\")
    if quote == '"""':
        if '"""' in body:
            raise WriterError("candidate contains triple quotes; cannot render safely")
    else:
        body = body.replace('"', '\\"')
    rendered = f"f{quote}{body}{quote}"

    parsed = cst.parse_expression(rendered)
    if not isinstance(parsed, cst.FormattedString):
        raise WriterError("rendered f-string failed to parse")
    if _template_of(parsed) != content:
        raise WriterError("rendered f-string failed round-trip verification")
    return rendered


def _template_of(node: cst.FormattedString) -> str:
    parts: list[str] = []
    module = cst.Module(body=[])
    for part in node.parts:
        if isinstance(part, cst.FormattedStringText):
            parts.append(part.value.replace("{{", "{").replace("}}", "}"))
        elif isinstance(part, cst.FormattedStringExpression):
            parts.append("{" + module.code_for_node(part.expression).strip() + "}")
    return "".join(parts)


class _NodeReplacer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, node_type: str, line: int, column: int, replacement: cst.BaseExpression):
        self._node_type = node_type
        self._line = line
        self._column = column
        self._replacement = replacement
        self.replaced = False
        # Nesting depth of the current `+` / adjacency concatenation chain, so
        # only the outermost node is replaced (a + b + c shares a start
        # position with its inner a + b).
        self._concat_depth = 0

    def _matches(self, node: cst.CSTNode) -> bool:
        position = self.get_metadata(PositionProvider, node).start
        return position.line == self._line and position.column == self._column

    def leave_SimpleString(
        self, original_node: cst.SimpleString, updated_node: cst.SimpleString
    ) -> cst.BaseExpression:
        if self._node_type == "SimpleString" and self._matches(original_node):
            self.replaced = True
            return self._replacement
        return updated_node

    def leave_FormattedString(
        self, original_node: cst.FormattedString, updated_node: cst.FormattedString
    ) -> cst.BaseExpression:
        if self._node_type == "FormattedString" and self._matches(original_node):
            self.replaced = True
            return self._replacement
        return updated_node

    def visit_ConcatenatedString(self, node: cst.ConcatenatedString) -> None:
        self._concat_depth += 1

    def leave_ConcatenatedString(
        self, original_node: cst.ConcatenatedString, updated_node: cst.BaseExpression
    ) -> cst.BaseExpression:
        self._concat_depth -= 1
        return self._maybe_replace_concat(original_node, updated_node)

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
        if isinstance(node.operator, cst.Add):
            self._concat_depth += 1

    def leave_BinaryOperation(
        self, original_node: cst.BinaryOperation, updated_node: cst.BaseExpression
    ) -> cst.BaseExpression:
        if not isinstance(original_node.operator, cst.Add):
            return updated_node
        self._concat_depth -= 1
        return self._maybe_replace_concat(original_node, updated_node)

    def _maybe_replace_concat(
        self, original_node: cst.CSTNode, updated_node: cst.BaseExpression
    ) -> cst.BaseExpression:
        # Replace only the outermost concatenation (depth back to zero) that
        # sits at the manifest's recorded position.
        if (
            self._node_type == "Concatenation"
            and self._concat_depth == 0
            and not self.replaced
            and self._matches(original_node)
        ):
            self.replaced = True
            return self._replacement
        return updated_node


class _ParamSetter(cst.CSTTransformer):
    """Set generation params (keyword args) on one model call, located by the
    prompt it sends or by the traced call-site line."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(
        self,
        prompt_key: str | None,
        call_site_line: int | None,
        overrides: dict[str, object],
    ) -> None:
        # Key off the first line only: a multi-line key breaks on CRLF-vs-LF,
        # since the parsed prompt uses \n while raw source may use \r\n.
        self._key = (prompt_key or "").splitlines()[0][:60] if prompt_key else ""
        self._line = call_site_line
        self._overrides = overrides
        self.changed = False
        self._done = False

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        if self._done:
            return updated_node
        matched = False
        if self._key:
            code = cst.Module(body=[]).code_for_node(original_node)
            matched = self._key in code
        if not matched and self._line is not None:
            start = self.get_metadata(PositionProvider, original_node).start
            matched = abs(start.line - self._line) <= 2
        if not matched:
            return updated_node
        return self._apply(updated_node)

    def _apply(self, call: cst.Call) -> cst.Call:
        import json

        args = list(call.args)
        for key, value in self._overrides.items():
            literal = json.dumps(value) if isinstance(value, str) else str(value)
            expr = cst.parse_expression(literal)
            index = next(
                (
                    i
                    for i, arg in enumerate(args)
                    if isinstance(arg.keyword, cst.Name) and arg.keyword.value == key
                ),
                None,
            )
            if index is not None:
                args[index] = args[index].with_changes(value=expr)
            else:
                if args:
                    args[-1] = args[-1].with_changes(comma=cst.Comma())
                args.append(
                    cst.Arg(
                        keyword=cst.Name(key),
                        value=expr,
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(""),
                            whitespace_after=cst.SimpleWhitespace(""),
                        ),
                    )
                )
            self.changed = True
        self._done = True
        return call.with_changes(args=args)


def set_call_params(
    source: str,
    prompt_snippet: str | None,
    call_site_line: int | None,
    param_overrides: dict[str, object],
) -> str | None:
    """Set generation params on the model call that sends ``prompt_snippet``.

    Deterministic AST edit (libcst) — no LLM, no git. Returns the rewritten
    source, or None when the call can't be located (caller falls back to the
    code delegate).
    """
    wrapper = MetadataWrapper(cst.parse_module(source))
    setter = _ParamSetter(prompt_snippet, call_site_line, param_overrides)
    updated = wrapper.visit(setter)
    return updated.code if setter.changed else None


def _replace_node(source: str, node_path: str, new_node_source: str) -> str:
    match = re.fullmatch(r"(\w+)@(\d+):(\d+)", node_path)
    if match is None:
        raise WriterError(f"malformed node path: {node_path!r}")
    node_type, line, column = match.group(1), int(match.group(2)), int(match.group(3))

    replacement = cst.parse_expression(new_node_source)
    wrapper = MetadataWrapper(cst.parse_module(source))
    replacer = _NodeReplacer(node_type, line, column, replacement)
    updated = wrapper.visit(replacer)
    if not replacer.replaced:
        raise WriterError(
            f"node {node_path} not found — the file changed since the manifest was built; "
            "run 'agentopt observe' to refresh it"
        )
    return updated.code
