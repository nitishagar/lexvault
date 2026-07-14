"""Configuration model for lexvault.

Phase 1: placeholder. The full pydantic ``LexVaultConfig`` model, the
``DictionaryTerm`` / ``RegexTerm`` types, and ``load_dictionary`` land in
Phase 2 (core masking engine).
"""

from __future__ import annotations


class LexVaultConfig:  # noqa: D101 — replaced in Phase 2.
    """Placeholder. Real pydantic model lands in Phase 2."""

    def __init__(self, **_kwargs: object) -> None:
        """Accept arbitrary kwargs so the skeleton imports without a config."""
