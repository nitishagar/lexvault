"""Shared pytest fixtures for lexvault tests.

Unit tests have ZERO LiteLLM dependency — they exercise the pure engine,
detector, vault, and streaming buffer invariants directly. The fixtures here
construct small in-memory/temp dictionaries + vaults so tests stay readable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lexvault.config import DictionaryTerm, RegexTerm, placeholder_namespace_regex
from lexvault.detector import Detector
from lexvault.vault import MappingVault

ORG_KEY = "test-org-key-keep-secret"
SCOPE = "default"
PLACEHOLDER_FORMAT = "[LEX-{code}]"


@pytest.fixture
def org_key() -> str:
    return ORG_KEY


@pytest.fixture
def scope() -> str:
    return SCOPE


@pytest.fixture
def placeholder_format() -> str:
    return PLACEHOLDER_FORMAT


@pytest.fixture
def namespace_re(placeholder_format: str) -> str:
    return placeholder_namespace_regex(placeholder_format)


@pytest.fixture
def dictionary() -> list[DictionaryTerm]:
    return [
        DictionaryTerm(term="Project Titan", type="codename"),
        DictionaryTerm(term="Titan", type="codename"),
        DictionaryTerm(term="customer_database", type="schema"),
    ]


@pytest.fixture
def regex_terms() -> list[RegexTerm]:
    return [RegexTerm(name="Employee ID", pattern=r"EMP-\d{4,6}", type="id")]


@pytest.fixture
def detector(
    dictionary: list[DictionaryTerm], regex_terms: list[RegexTerm], namespace_re: str
) -> Detector:
    return Detector(dictionary, regex_terms, namespace_re)


@pytest.fixture
def vault(tmp_path: Path) -> MappingVault:
    """A fresh vault in a per-test temp directory (pytest-managed cleanup)."""
    return MappingVault(tmp_path / "vault.db")
