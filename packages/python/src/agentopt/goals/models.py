"""Typed view of a parsed GOALS.md (format: ``schemas/goals-format.md``)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

RECOGNIZED_DIMENSIONS = ("accuracy", "cost", "speed")


class Rule(BaseModel):
    """A behavior that must always hold; compiled into a hard check."""

    model_config = ConfigDict(extra="forbid")

    text: str
    surface: str | None = None


class Example(BaseModel):
    """An exact input → expectation pair; compiled into an example check."""

    model_config = ConfigDict(extra="forbid")

    input: str
    expectation: str
    surface: str | None = None


class Quality(BaseModel):
    """A fuzzy behavior graded by the judge, anchored by examples."""

    model_config = ConfigDict(extra="forbid")

    text: str
    good_examples: list[str] = Field(default_factory=list)
    bad_examples: list[str] = Field(default_factory=list)
    surface: str | None = None


class GlossaryItem(BaseModel):
    """A domain term the judge must understand."""

    model_config = ConfigDict(extra="forbid")

    term: str
    definition: str


class Goals(BaseModel):
    """Everything stated in one GOALS.md."""

    model_config = ConfigDict(extra="forbid")

    rules: list[Rule] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    qualities: list[Quality] = Field(default_factory=list)
    priorities: list[str] = Field(default_factory=lambda: list(RECOGNIZED_DIMENSIONS))
    glossary: list[GlossaryItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.rules or self.examples or self.qualities)
