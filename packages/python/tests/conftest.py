"""Shared fixtures: repo-level schema files and example artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"
SCHEMAS_DIR = TESTS_DIR.parents[2] / "schemas"


@pytest.fixture(scope="session")
def schemas_dir() -> Path:
    return SCHEMAS_DIR


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


def load_schema(name: str) -> dict[str, Any]:
    schema: dict[str, Any] = json.loads((SCHEMAS_DIR / name).read_text(encoding="utf-8"))
    return schema
