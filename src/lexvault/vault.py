"""Local mapping vault for lexvault.

SQLite-backed store of (scope, placeholder → original) mappings. The vault is
the **cross-hook state store** (IMPLICIT_SPEC invariant 3): ``async_pre_call_hook``
writes the mask↔original mapping, and both restore hooks look it up by
placeholder string. NO mappings live in guardrail instance attributes (that
would race under concurrent requests — the Presidio bug).

Concurrency: WAL journal mode + ``busy_timeout=5000`` + a single
``asyncio.Lock`` around writes (invariant 13). Reads are lock-free under WAL.
All public methods are ``async`` (the guardrail is async); a synchronous helper
exists for unit tests.

Security: the directory is created ``0700`` and the vault file ``0600`` because
the vault contains the original (unmasked) terms (invariant 12). No network
code. No TTL/eviction in v0.1 (invariant 14).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["MappingVault", "VaultError"]


class VaultError(RuntimeError):
    """Raised on vault failure. Callers fail-closed (invariant 11)."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS mappings (
    scope        TEXT    NOT NULL,
    placeholder  TEXT    NOT NULL,
    original     TEXT    NOT NULL,
    term_type    TEXT,
    created_at   TEXT    NOT NULL,
    request_id   TEXT,
    PRIMARY KEY (placeholder)
);
CREATE INDEX IF NOT EXISTS idx_mappings_scope_original ON mappings(scope, original);
"""

# Module-level default path sentinel — avoids a function-call in a default arg.
_DEFAULT_VAULT_PATH = Path.home() / ".lexvault" / "vault.db"


class MappingVault:
    """SQLite mapping vault: placeholder → original, keyed by scope.

    ``assign`` records a deterministic placeholder for an original term. If the
    placeholder is already taken by a *different* original (a truncated-HMAC
    collision), a ``-2``, ``-3`` … disambiguator is appended and the resulting
    placeholder is returned. ``lookup`` retrieves the original for a placeholder
    string. Both are idempotent under repeated identical input.
    """

    def __init__(self, vault_path: Path | str | None = None) -> None:
        self._path = Path(vault_path) if vault_path is not None else _DEFAULT_VAULT_PATH
        self._lock = asyncio.Lock()
        self._conn = self._open(self._path)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _open(path: Path) -> sqlite3.Connection:
        """Open the vault file, creating its directory with restrictive modes."""
        path = Path(path)
        # Create parent dir 0700; create the db file 0600 via a first touch.
        path.parent.mkdir(parents=True, exist_ok=True)
        # On some filesystems chmod may be restricted (EPERM); the mkdir mode
        # is the primary guard, so don't fail open over a chmod EPERM.
        with contextlib.suppress(PermissionError):
            path.parent.chmod(0o700)
        # Touch the file with 0600 before SQLite opens it (SQLite creates with
        # default umask otherwise). Use a secure tempfile-style open so we own
        # the mode atomically.
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
        else:
            # Tighten an existing file's mode defensively.
            with contextlib.suppress(PermissionError):
                path.chmod(0o600)

        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL: readers don't block writers; busy_timeout: bounded wait under
        # contention (invariant 13).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        # Seed schema_version if empty.
        row = conn.execute("SELECT COUNT(*) AS c FROM schema_version").fetchone()
        if row["c"] == 0:
            conn.execute("INSERT INTO schema_version(version) VALUES (1)")
        conn.commit()
        return conn

    # ------------------------------------------------------------------ #
    # async public API (used by the guardrail)
    # ------------------------------------------------------------------ #
    async def assign(
        self,
        scope: str,
        placeholder: str,
        original: str,
        *,
        request_id: str | None,
        term_type: str | None = None,
    ) -> str:
        """Record ``placeholder → original`` for ``scope``. Returns the final placeholder.

        If ``placeholder`` is already mapped to ``original`` (idempotent re-mask),
        it's a no-op and the same placeholder is returned. If it's mapped to a
        *different* original (HMAC collision), a ``-N`` disambiguator is appended
        until a free slot is found (fail-closed if exhausted). The write is
        serialized under the vault ``asyncio.Lock`` (invariant 13).
        """
        async with self._lock:
            return self._assign_locked(
                scope, placeholder, original, request_id=request_id, term_type=term_type
            )

    async def lookup(self, placeholder: str) -> str | None:
        """Return the original for ``placeholder`` (lock-free WAL read)."""
        return self._lookup_sync(placeholder)

    async def close(self) -> None:
        async with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # sync internals (also exposed for unit tests)
    # ------------------------------------------------------------------ #
    def _assign_locked(
        self,
        scope: str,
        placeholder: str,
        original: str,
        *,
        request_id: str | None,
        term_type: str | None,
    ) -> str:
        candidate = placeholder
        now = datetime.now(timezone.utc).isoformat()
        # Bounded collision loop: deterministic placeholders rarely collide;
        # cap at a sane upper bound and fail-closed if exhausted.
        for n in range(2, 1000):
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO mappings(scope, placeholder, original, term_type, created_at, request_id) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (scope, candidate, original, term_type, now, request_id),
                    )
            except sqlite3.DatabaseError as exc:  # pragma: no cover - DB corruption
                msg = f"vault write failed: {exc}"
                raise VaultError(msg) from exc

            mapped = self._lookup_sync(candidate)
            if mapped == original:
                return candidate
            # Slot holds a different original → collision: append disambiguator.
            candidate = f"{placeholder}-{n}"
        msg = f"placeholder collision exhausted for {placeholder!r}"
        raise VaultError(msg)

    def _lookup_sync(self, placeholder: str) -> str | None:
        try:
            row = self._conn.execute(
                "SELECT original FROM mappings WHERE placeholder = ?", (placeholder,)
            ).fetchone()
        except sqlite3.DatabaseError as exc:  # pragma: no cover - DB corruption
            msg = f"vault read failed: {exc}"
            raise VaultError(msg) from exc
        return None if row is None else row["original"]


def temp_vault() -> tuple[MappingVault, Path]:
    """Construct a vault in a fresh temp directory (for tests)."""
    d = Path(tempfile.mkdtemp(prefix="lexvault-test-"))
    path = d / "vault.db"
    return MappingVault(path), path
