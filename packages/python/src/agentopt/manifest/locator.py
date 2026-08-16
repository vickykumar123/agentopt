"""Static location of prompt text in Python source, via libcst.

Runtime strings from traces are matched against string nodes in the user's
files: plain literals match exactly; f-strings match with their interpolated
expressions treated as wildcard slots. libcst preserves exact formatting, so a
located node can later be rewritten without touching anything around it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

from agentopt.pattern import match_template_pattern

logger = logging.getLogger("agentopt.manifest")

MIN_STATIC_MATCH_LENGTH = 8
"""An f-string must contain at least this much static text to be matchable;
below it (e.g. ``f"{x}"``) almost anything would match."""

_SCAN_EXCLUDED_DIRS = frozenset(
    {".venv", "venv", "site-packages", "node_modules", ".agentopt", ".git", "__pycache__"}
)


def _concat_leaves(node: cst.BaseExpression) -> list[cst.BaseExpression]:
    """Flatten a string concatenation (adjacency or ``+``) into its operands."""
    if isinstance(node, cst.ConcatenatedString):
        return _concat_leaves(node.left) + _concat_leaves(node.right)
    if isinstance(node, cst.BinaryOperation) and isinstance(node.operator, cst.Add):
        return _concat_leaves(node.left) + _concat_leaves(node.right)
    return [node]


@dataclass
class StringCandidate:
    """One string node in one file that a runtime prompt could originate from."""

    file: str
    line: int
    column: int
    node_type: str
    source: str
    static_value: str | None = None
    pattern: str | None = None
    slots: list[str] = field(default_factory=list)
    template: str = ""

    @property
    def node_path(self) -> str:
        return f"{self.node_type}@{self.line}:{self.column}"

    def matches(self, content: str) -> bool:
        if self.static_value is not None:
            return self.static_value == content
        if self.pattern is not None:
            return match_template_pattern(self.pattern, content) is not None
        return False


class _StringCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, module: cst.Module, file: str) -> None:
        self._module = module
        self._file = file
        self.candidates: list[StringCandidate] = []

    def visit_SimpleString(self, node: cst.SimpleString) -> None:
        value = node.evaluated_value
        if not isinstance(value, str) or not value:
            return
        self.candidates.append(
            self._candidate(node, "SimpleString", static_value=value),
        )

    def visit_FormattedString(self, node: cst.FormattedString) -> None:
        candidate = self._parts_to_candidate(node, self._leaf_parts(node), "FormattedString")
        if candidate is not None:
            self.candidates.append(candidate)

    def visit_ConcatenatedString(self, node: cst.ConcatenatedString) -> bool:
        # Implicit adjacency ("a" "b") is one prompt; its fragments are not
        # independently editable, so do not descend into them.
        self._handle_concat(node)
        return False

    def visit_BinaryOperation(self, node: cst.BinaryOperation) -> bool:
        # Only `+` chains that contain a string literal are string
        # concatenation; everything else (arithmetic) descends normally.
        if not isinstance(node.operator, cst.Add):
            return True
        leaves = _concat_leaves(node)
        if not any(isinstance(leaf, (cst.SimpleString, cst.FormattedString)) for leaf in leaves):
            return True
        self._handle_concat(node)
        return False

    def _handle_concat(self, node: cst.BaseExpression) -> None:
        """Constant-fold a concatenation into one candidate.

        Static literals become fixed text; non-literal operands become
        ``{slot}`` placeholders — the same representation as an f-string, so a
        concatenated prompt is both matchable at runtime and rewritable.
        """
        parts: list[tuple[str, str]] = []
        for leaf in _concat_leaves(node):
            parts.extend(self._leaf_parts(leaf))
        candidate = self._parts_to_candidate(node, parts, "Concatenation")
        if candidate is not None:
            self.candidates.append(candidate)

    def _leaf_parts(self, leaf: cst.BaseExpression) -> list[tuple[str, str]]:
        """One operand as ordered ``("static", text)`` / ``("slot", expr)`` pieces."""
        if isinstance(leaf, cst.SimpleString):
            value = leaf.evaluated_value
            if isinstance(value, str):
                return [("static", value)]
            return [("slot", self._module.code_for_node(leaf).strip())]
        if isinstance(leaf, cst.FormattedString):
            parts: list[tuple[str, str]] = []
            for part in leaf.parts:
                if isinstance(part, cst.FormattedStringText):
                    parts.append(("static", part.value.replace("{{", "{").replace("}}", "}")))
                elif isinstance(part, cst.FormattedStringExpression):
                    parts.append(("slot", self._module.code_for_node(part.expression).strip()))
            return parts
        if isinstance(leaf, (cst.Integer, cst.Float)):
            return [("static", self._module.code_for_node(leaf).strip())]
        return [("slot", self._module.code_for_node(leaf).strip())]

    def _parts_to_candidate(
        self,
        node: cst.CSTNode,
        parts: list[tuple[str, str]],
        node_type: str,
    ) -> StringCandidate | None:
        pattern_parts: list[str] = []
        template_parts: list[str] = []
        slots: list[str] = []
        static_length = 0
        for kind, value in parts:
            if kind == "static":
                pattern_parts.append(re.escape(value))
                template_parts.append(value)
                static_length += len(value)
            else:
                slots.append(value)
                pattern_parts.append("(.*?)")
                template_parts.append("{" + value + "}")
        if not slots:
            folded = "".join(template_parts)
            if not folded:
                return None
            return self._candidate(node, node_type, static_value=folded)
        # Below the static floor almost anything matches — not a reliable anchor.
        if static_length < MIN_STATIC_MATCH_LENGTH:
            return None
        return self._candidate(
            node,
            node_type,
            pattern="".join(pattern_parts),
            slots=slots,
            template="".join(template_parts),
        )

    def _candidate(
        self,
        node: cst.CSTNode,
        node_type: str,
        static_value: str | None = None,
        pattern: str | None = None,
        slots: list[str] | None = None,
        template: str | None = None,
    ) -> StringCandidate:
        position = self.get_metadata(PositionProvider, node).start
        return StringCandidate(
            file=self._file,
            line=position.line,
            column=position.column,
            node_type=node_type,
            source=self._module.code_for_node(node),
            static_value=static_value,
            pattern=pattern,
            slots=slots or [],
            template=template if template is not None else (static_value or ""),
        )


class PromptLocator:
    """Indexes a project's Python files and matches runtime strings to nodes."""

    def __init__(self, project_root: Path, exclude_globs: tuple[str, ...] = ()) -> None:
        self._project_root = project_root.resolve()
        self._exclude_globs = exclude_globs
        self._index: dict[str, list[StringCandidate]] = {}

    def _is_excluded(self, relative_file: str) -> bool:
        import fnmatch

        return any(fnmatch.fnmatch(relative_file, glob) for glob in self._exclude_globs)

    def candidates_in(self, relative_file: str) -> list[StringCandidate]:
        """String candidates of one file (indexed on first access)."""
        if self._is_excluded(relative_file):
            return []
        if relative_file not in self._index:
            self._index[relative_file] = self._index_file(relative_file)
        return self._index[relative_file]

    def find(self, content: str, preferred_file: str | None) -> list[StringCandidate]:
        """All candidates matching ``content``.

        If any match lives in ``preferred_file`` (the call site), matches are
        narrowed to that file — the closest plausible origin wins.
        """
        if preferred_file is not None:
            local = [c for c in self.candidates_in(preferred_file) if c.matches(content)]
            if local:
                return local
        matches: list[StringCandidate] = []
        for file in self._project_files():
            matches.extend(c for c in self.candidates_in(file) if c.matches(content))
        return matches

    def _index_file(self, relative_file: str) -> list[StringCandidate]:
        path = self._project_root / relative_file
        try:
            source = path.read_text(encoding="utf-8")
            wrapper = MetadataWrapper(cst.parse_module(source))
        except (OSError, cst.ParserSyntaxError, UnicodeDecodeError):
            logger.debug("cannot index %s", relative_file, exc_info=True)
            return []
        module = wrapper.module
        collector = _StringCollector(module, relative_file)
        wrapper.visit(collector)
        return collector.candidates

    def _project_files(self) -> list[str]:
        files = []
        for path in sorted(self._project_root.rglob("*.py")):
            if not _SCAN_EXCLUDED_DIRS.isdisjoint(path.parts):
                continue
            relative = path.relative_to(self._project_root).as_posix()
            if not self._is_excluded(relative):
                files.append(relative)
        return files

    def candidate_at(self, file: str, node_path: str) -> StringCandidate | None:
        """The candidate a manifest entry points at, re-located by node path."""
        for candidate in self.candidates_in(file):
            if candidate.node_path == node_path:
                return candidate
        return None
