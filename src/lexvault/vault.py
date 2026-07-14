"""Local mapping vault for lexvault.

Phase 1: placeholder. The real SQLite-backed ``MappingVault`` (WAL mode,
busy_timeout, asyncio.Lock around writes, INSERT OR IGNORE + collision suffix,
lookup-by-placeholder) lands in Phase 2.
"""

from __future__ import annotations


class MappingVault:  # noqa: D101 — replaced in Phase 2.
    """Placeholder. Real SQLite-backed vault lands in Phase 2."""

    def __init__(self, **_kwargs: object) -> None:
        """Accept arbitrary kwargs so the skeleton imports without a config."""
