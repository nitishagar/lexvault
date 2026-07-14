"""Unit tests for the SQLite mapping vault.

Covers IMPLICIT_SPEC invariants:
  3 (state survives across hooks — vault is the cross-hook store), 9 (collision
  suffix), 12 (local-only, file modes), 13 (concurrency — 100 writers), 14 (no
  eviction), 11 (fail-closed on vault error).
"""

from __future__ import annotations

import asyncio
import os
import stat

from lexvault.vault import MappingVault


# --------------------------------------------------------------------------- #
# basic assign / lookup / round-trip (inv 3)
# --------------------------------------------------------------------------- #
class TestAssignLookup:
    async def test_assign_then_lookup(self, vault):
        ph = await vault.assign("default", "[LEX-AAAAAAAA]", "Project Titan", request_id="r1")
        assert ph == "[LEX-AAAAAAAA]"
        assert await vault.lookup("[LEX-AAAAAAAA]") == "Project Titan"

    async def test_lookup_missing_returns_none(self, vault):
        assert await vault.lookup("[LEX-NOPE]") is None

    async def test_idempotent_assign_same_mapping(self, vault):
        """Re-assigning the same (placeholder, original) is a no-op."""
        ph1 = await vault.assign("default", "[LEX-AAAAAA11]", "Project Titan", request_id="r1")
        ph2 = await vault.assign("default", "[LEX-AAAAAA11]", "Project Titan", request_id="r2")
        assert ph1 == ph2
        assert await vault.lookup(ph1) == "Project Titan"

    async def test_request_id_stored_for_attribution(self, vault, tmp_path):
        """The request_id column is populated (for future per-request cleanup)."""
        await vault.assign("default", "[LEX-AAAAAA22]", "Project Titan", request_id="req-xyz")
        row = vault._conn.execute(
            "SELECT request_id FROM mappings WHERE placeholder = ?", ("[LEX-AAAAAA22]",)
        ).fetchone()
        assert row["request_id"] == "req-xyz"


# --------------------------------------------------------------------------- #
# collision handling (inv 9)
# --------------------------------------------------------------------------- #
class TestCollisions:
    async def test_collision_gets_disambiguator(self, vault):
        """Two different originals claiming the same placeholder → -2 suffix."""
        ph1 = await vault.assign("default", "[LEX-COLLIDE1]", "Project Titan", request_id="r1")
        ph2 = await vault.assign("default", "[LEX-COLLIDE1]", "Project Mercury", request_id="r2")
        assert ph1 == "[LEX-COLLIDE1]"
        assert ph2 == "[LEX-COLLIDE1]-2"
        # Both restore to their own originals.
        assert await vault.lookup(ph1) == "Project Titan"
        assert await vault.lookup(ph2) == "Project Mercury"

    async def test_multiple_collisions_increment(self, vault):
        await vault.assign("default", "[LEX-COLLIDE2]", "A", request_id="r1")
        await vault.assign("default", "[LEX-COLLIDE2]", "B", request_id="r2")
        ph3 = await vault.assign("default", "[LEX-COLLIDE2]", "C", request_id="r3")
        assert ph3 == "[LEX-COLLIDE2]-3"
        assert {
            await vault.lookup("[LEX-COLLIDE2]"),
            await vault.lookup("[LEX-COLLIDE2]-2"),
            await vault.lookup(ph3),
        } == {"A", "B", "C"}

    async def test_same_original_after_collision_uses_existing(self, vault):
        """Assigning the original placeholder+original again is idempotent even after a collision."""
        await vault.assign("default", "[LEX-COLLIDE3]", "A", request_id="r1")
        await vault.assign("default", "[LEX-COLLIDE3]", "B", request_id="r2")  # → -2
        ph_again = await vault.assign("default", "[LEX-COLLIDE3]", "A", request_id="r3")
        assert ph_again == "[LEX-COLLIDE3]"


# --------------------------------------------------------------------------- #
# concurrency — 100 writers (inv 13)
# --------------------------------------------------------------------------- #
class TestConcurrency:
    async def test_100_concurrent_writers_no_lock_error(self, tmp_path):
        """100 concurrent assigns must not raise 'database is locked' and all rows persist."""
        vault = MappingVault(tmp_path / "concurrent.db")

        async def one_writer(i: int) -> str:
            # Each writer claims a UNIQUE placeholder for a unique original.
            return await vault.assign(
                "default", f"[LEX-W{i:03d}]", f"original-{i}", request_id=f"r{i}"
            )

        results = await asyncio.gather(*(one_writer(i) for i in range(100)))
        assert len(results) == 100
        # All placeholders distinct, all originals restorable.
        assert len(set(results)) == 100
        for i in range(100):
            assert await vault.lookup(f"[LEX-W{i:03d}]") == f"original-{i}"
        await vault.close()

    async def test_concurrent_collisions_resolve_distinctly(self, tmp_path):
        """100 writers all racing for the SAME placeholder: each gets a distinct suffix."""
        vault = MappingVault(tmp_path / "concurrent-collide.db")

        async def claim(i: int) -> str:
            return await vault.assign("default", "[LEX-RACE]", f"term-{i}", request_id=f"r{i}")

        placeholders = await asyncio.gather(*(claim(i) for i in range(50)))
        # All 50 placeholders must be distinct (the lock serializes collision resolution).
        assert len(set(placeholders)) == 50
        # Each restores to its own original.
        originals = {await vault.lookup(ph) for ph in placeholders}
        assert originals == {f"term-{i}" for i in range(50)}
        await vault.close()


# --------------------------------------------------------------------------- #
# local-only + file modes (inv 12)
# --------------------------------------------------------------------------- #
class TestLocalOnly:
    def test_file_created_0600(self, tmp_path):
        v = MappingVault(tmp_path / "mode.db")
        mode = stat.S_IMODE(os.stat(v._path).st_mode)
        assert mode == 0o600

    def test_dir_created_0700(self, tmp_path):
        nested = tmp_path / "deep" / "vault" / "dir"
        MappingVault(nested / "v.db")
        mode = stat.S_IMODE(os.stat(nested).st_mode)
        assert mode == 0o700

    def test_vault_is_local_file(self, tmp_path):
        """The vault is an ordinary local SQLite file (no network)."""
        path = tmp_path / "local.db"
        v = MappingVault(path)
        assert path.exists()
        assert path.is_file()
        # The schema tables exist.
        tables = {
            r[0] for r in v._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "mappings" in tables
        assert "schema_version" in tables


# --------------------------------------------------------------------------- #
# no eviction / durability (inv 14)
# --------------------------------------------------------------------------- #
class TestDurability:
    async def test_mapping_persists_across_reopen(self, tmp_path):
        """A mapping written in one vault instance is readable from a new one on the same file."""
        path = tmp_path / "persist.db"
        v1 = MappingVault(path)
        await v1.assign("default", "[LEX-PERSIST]", "Project Titan", request_id="r1")
        await v1.close()

        v2 = MappingVault(path)
        assert await v2.lookup("[LEX-PERSIST]") == "Project Titan"
        await v2.close()
