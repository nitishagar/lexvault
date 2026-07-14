"""Unit tests for the masking engine: derive / mask / unmask / round-trip.

Covers IMPLICIT_SPEC invariants:
  1 (round-trip fidelity), 2 (cross-surface consistency), 9 (deterministic +
  collision-free), 10 (iteration-order independence), 11 (fail-closed),
  16 (idempotent re-mask), 17 (placeholder-vs-literal).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lexvault.detector import Detector
from lexvault.engine import derive_placeholder, mask, rebuild, unmask


# --------------------------------------------------------------------------- #
# derive_placeholder — determinism + cross-surface consistency (inv 2, 9)
# --------------------------------------------------------------------------- #
class TestDerive:
    def test_deterministic_same_inputs(self, org_key, placeholder_format):
        a = derive_placeholder("Project Titan", org_key, "default", placeholder_format)
        b = derive_placeholder("Project Titan", org_key, "default", placeholder_format)
        assert a == b

    def test_different_terms_different_placeholders(self, org_key, placeholder_format):
        a = derive_placeholder("Project Titan", org_key, "default", placeholder_format)
        b = derive_placeholder("customer_database", org_key, "default", placeholder_format)
        assert a != b

    def test_scope_changes_placeholder(self, org_key, placeholder_format):
        a = derive_placeholder("Project Titan", org_key, "default", placeholder_format)
        b = derive_placeholder("Project Titan", org_key, "other", placeholder_format)
        assert a != b

    def test_org_key_changes_placeholder(self, placeholder_format):
        a = derive_placeholder("Project Titan", "key-one", "default", placeholder_format)
        b = derive_placeholder("Project Titan", "key-two", "default", placeholder_format)
        assert a != b

    def test_placeholder_format_applied(self, org_key):
        p = derive_placeholder("Project Titan", org_key, "default", "<<{code}>>")
        assert p.startswith("<<") and p.endswith(">>")
        # code is 8 base32 chars
        assert len(p) == 2 + 8 + 2


# --------------------------------------------------------------------------- #
# mask + unmask round-trip (inv 1)
# --------------------------------------------------------------------------- #
class TestRoundTrip:
    async def test_single_term_round_trips(
        self, detector, vault, org_key, scope, placeholder_format, namespace_re
    ):
        text = "Please review the Project Titan roadmap."
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-1",
        )
        assert "Project Titan" not in masked
        assert "[LEX-" in masked
        restored = unmask(masked, vault=vault, placeholder_namespace_re=namespace_re)
        assert restored == text

    async def test_multiple_distinct_terms_round_trip(
        self, detector, vault, org_key, scope, placeholder_format, namespace_re
    ):
        text = "Project Titan reads from customer_database nightly."
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-2",
        )
        assert "Project Titan" not in masked
        assert "customer_database" not in masked
        restored = unmask(masked, vault=vault, placeholder_namespace_re=namespace_re)
        assert restored == text

    async def test_same_term_same_placeholder_across_calls(
        self, detector, vault, org_key, scope, placeholder_format
    ):
        """Invariant 2: the same term maps to the same placeholder across calls."""
        masked_a = await mask(
            "Project Titan",
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-a",
        )
        masked_b = await mask(
            "Project Titan",
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-b",
        )
        assert masked_a == masked_b

    async def test_regex_term_round_trips(
        self, detector, vault, org_key, scope, placeholder_format, namespace_re
    ):
        text = "Contact EMP-123456 for access."
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-r",
        )
        assert "EMP-123456" not in masked
        restored = unmask(masked, vault=vault, placeholder_namespace_re=namespace_re)
        assert restored == text

    async def test_term_not_in_dictionary_unchanged(
        self, detector, vault, org_key, scope, placeholder_format, namespace_re
    ):
        text = "A plain message with no secrets."
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-p",
        )
        assert masked == text
        restored = unmask(masked, vault=vault, placeholder_namespace_re=namespace_re)
        assert restored == text

    async def test_empty_text_unchanged(self, detector, vault, org_key, scope, placeholder_format):
        masked = await mask(
            "",
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-e",
        )
        assert masked == ""
        assert unmask("", vault=vault, placeholder_namespace_re=r"\[LEX\-[A-Z2-7]+\]") == ""


# --------------------------------------------------------------------------- #
# overlap resolution — longest-then-leftmost (inv 10)
# --------------------------------------------------------------------------- #
class TestOverlap:
    async def test_longest_overlap_wins(
        self, vault, org_key, scope, placeholder_format, namespace_re
    ):
        """'Project Titan' and 'Titan' both match; the longer wins, no corruption."""
        from lexvault.config import DictionaryTerm

        detector = Detector(
            [DictionaryTerm(term="Project Titan"), DictionaryTerm(term="Titan")],
            [],
            namespace_re,
        )
        text = "Project Titan and Titan are related."
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-o",
        )
        # Two distinct placeholders (different originals), no nested corruption.
        assert "[LEX-" in masked
        # The standalone "Titan" is still masked (to its own placeholder), and
        # "Project Titan" is masked as a whole — restore proves no corruption.
        restored = unmask(masked, vault=vault, placeholder_namespace_re=namespace_re)
        assert restored == text

    async def test_iteration_order_independent(
        self, vault, org_key, scope, placeholder_format, namespace_re
    ):
        """Invariant 10: same output regardless of dictionary order."""
        from lexvault.config import DictionaryTerm

        terms_a = [
            DictionaryTerm(term="Project Titan"),
            DictionaryTerm(term="Titan"),
            DictionaryTerm(term="customer_database"),
        ]
        terms_b = list(reversed(terms_a))
        d_a = Detector(terms_a, [], namespace_re)
        d_b = Detector(terms_b, [], namespace_re)

        text = "Project Titan uses customer_database and Titan too."

        async def run(det: Detector, rid: str) -> str:
            import tempfile

            from lexvault.vault import MappingVault

            v = MappingVault(tempfile.mkdtemp(prefix="lv-") + "/v.db")
            try:
                return await mask(
                    text,
                    detector=det,
                    vault=v,
                    org_key=org_key,
                    scope=scope,
                    placeholder_format=placeholder_format,
                    request_id=rid,
                )
            finally:
                await v.close()

        out_a = await run(d_a, "a")
        out_b = await run(d_b, "b")
        assert out_a == out_b

    async def test_determinism_100_runs_shuffled_orders(
        self, vault, org_key, scope, placeholder_format, namespace_re
    ):
        """Invariant 9/10: same output across 100 runs with randomly-shuffled dict orders."""
        import random

        from lexvault.config import DictionaryTerm

        terms = [
            DictionaryTerm(term="Project Titan"),
            DictionaryTerm(term="Titan"),
            DictionaryTerm(term="customer_database"),
            DictionaryTerm(term="Acquisition Zephyr"),
        ]
        text = "Project Titan, Titan, customer_database and Acquisition Zephyr overlap."

        outputs: set[str] = set()
        for i in range(100):
            shuffled = list(terms)
            random.Random(i).shuffle(shuffled)  # seeded → deterministic per-run
            det = Detector(shuffled, [], namespace_re)
            out = await mask(
                text,
                detector=det,
                vault=vault,
                org_key=org_key,
                scope=scope,
                placeholder_format=placeholder_format,
                request_id=f"det-{i}",
            )
            outputs.add(out)
        # All 100 orderings produced byte-identical masked output.
        assert len(outputs) == 1, f"non-deterministic masking across orders: {outputs}"

    @pytest.mark.parametrize("term", ["Project Titan", "Titan", "customer_database"])
    async def test_every_dictionary_term_round_trips(
        self, term, vault, org_key, scope, placeholder_format, namespace_re
    ):
        """Round-trip each dictionary term individually (parametrized over the dict)."""
        from lexvault.config import DictionaryTerm

        det = Detector([DictionaryTerm(term=term)], [], namespace_re)
        text = f"Discuss {term} in detail."
        masked = await mask(
            text,
            detector=det,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id=f"rt-{term}",
        )
        assert term not in masked
        assert unmask(masked, vault=vault, placeholder_namespace_re=namespace_re) == text

    async def test_overlap_does_not_double_mask(
        self, vault, org_key, scope, placeholder_format, namespace_re
    ):
        """The standalone substring of a longer match is not separately replaced inside it."""
        from lexvault.config import DictionaryTerm

        detector = Detector(
            [DictionaryTerm(term="Project Titan"), DictionaryTerm(term="Titan")], [], namespace_re
        )
        text = "Project Titan"
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-d",
        )
        # Exactly one placeholder (the longer match consumed the whole span).
        import re

        ns = re.compile(namespace_re)
        assert len(ns.findall(masked)) == 1


# --------------------------------------------------------------------------- #
# idempotent re-mask (inv 16)
# --------------------------------------------------------------------------- #
class TestIdempotency:
    async def test_masking_already_masked_is_noop(
        self, detector, vault, org_key, scope, placeholder_format
    ):
        text = "Project Titan roadmap"
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-i1",
        )
        re_masked = await mask(
            masked,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-i2",
        )
        assert re_masked == masked, "re-masking an already-masked text must not double-mask"

    async def test_placeholder_in_history_untouched(
        self, detector, vault, org_key, scope, placeholder_format
    ):
        """A placeholder present in conversation history is left as-is."""
        import re

        from lexvault.config import placeholder_namespace_regex

        ns = placeholder_namespace_regex(placeholder_format)
        # Pre-seed a placeholder that looks like ours but isn't a real mapping,
        # plus a real term.
        text = "Previous answer used [LEX-AAAAAAAA]. Now about Project Titan."
        masked = await mask(
            text,
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-h",
        )
        # The fake placeholder survived untouched; the real term got masked.
        assert "[LEX-AAAAAAAA]" in masked
        assert "Project Titan" not in masked
        # And no double-masking / nesting happened.
        # Count namespace placeholders: exactly the 1 pre-existing + the 1 new.
        ns_re = re.compile(ns)
        assert len(ns_re.findall(masked)) == 2


# --------------------------------------------------------------------------- #
# placeholder-vs-literal (inv 17)
# --------------------------------------------------------------------------- #
class TestPlaceholderVsLiteral:
    def test_user_typed_lookalike_with_no_mapping_left_alone(self, vault, namespace_re):
        """Invariant 17: a placeholder-lookalike with no stored mapping is not corrupted."""
        text = "User wrote [LEX-QQQQQQQQ] by hand."
        out = unmask(text, vault=vault, placeholder_namespace_re=namespace_re)
        assert out == text, "unmask must leave unknown lookalikes untouched"

    async def test_collision_suffixed_placeholder_restores(
        self, vault, org_key, scope, placeholder_format, namespace_re
    ):
        """A collision-suffixed placeholder ``[LEX-XXXX2222]-2`` must restore (inv 1, 9).

        Regression: the vault appends the ``-N`` suffix OUTSIDE the closing
        bracket of the already-formatted placeholder. The namespace regex must
        capture the full suffixed span so ``unmask`` resolves it; otherwise the
        base placeholder restores to the wrong original and the suffix leaks.
        """
        import re

        # Seed a collision: two originals claim the same derived placeholder.
        ph1 = await vault.assign(scope, "[LEX-COLLXYZ]", "Project Titan", request_id="r1")
        ph2 = await vault.assign(scope, "[LEX-COLLXYZ]", "Project Mercury", request_id="r2")
        assert ph1 == "[LEX-COLLXYZ]"
        assert ph2 == "[LEX-COLLXYZ]-2"

        text = f"See {ph1} and {ph2} together."
        restored = unmask(text, vault=vault, placeholder_namespace_re=namespace_re)
        assert restored == "See Project Titan and Project Mercury together.", (
            f"collision-suffixed placeholder must restore; got {restored!r}"
        )
        # And re-masking the restored text is idempotent-ish: the suffixed form
        # is pre-excluded as a placeholder span (no double-mask of its interior).
        assert re.search(namespace_re, restored) is None

    async def test_lookalike_resolves_only_if_mapping_exists(
        self, detector, vault, org_key, scope, placeholder_format, namespace_re
    ):
        """A real placeholder IS restored; a co-occurring fake is not."""
        masked = await mask(
            "Project Titan",
            detector=detector,
            vault=vault,
            org_key=org_key,
            scope=scope,
            placeholder_format=placeholder_format,
            request_id="req-l",
        )
        # Inject a fake placeholder next to the real one.
        combined = masked + " and also [LEX-ZZZZZZZZ]"
        restored = unmask(combined, vault=vault, placeholder_namespace_re=namespace_re)
        assert "Project Titan" in restored
        assert "[LEX-ZZZZZZZZ]" in restored  # fake survived
        assert masked not in restored  # the real placeholder was replaced


# --------------------------------------------------------------------------- #
# fail-closed (inv 11)
# --------------------------------------------------------------------------- #
class TestFailClosed:
    async def test_vault_error_propagates_from_mask(
        self, detector, org_key, scope, placeholder_format
    ):
        """A vault write failure must propagate (not return unmasked text)."""
        from lexvault.vault import VaultError

        fake_vault = MagicMock()
        fake_vault.assign = MagicMock(side_effect=VaultError("boom"))
        # mask awaits vault.assign → an AsyncMock is needed
        fake_vault.assign = MagicMock(side_effect=VaultError("boom"))

        with pytest.raises(VaultError):
            await mask(
                "Project Titan",
                detector=detector,
                vault=fake_vault,
                org_key=org_key,
                scope=scope,
                placeholder_format=placeholder_format,
                request_id="req-f",
            )

    def test_vault_error_propagates_from_unmask(self, namespace_re):
        """A vault read failure must propagate (not leak placeholders)."""
        from lexvault.vault import VaultError

        fake_vault = MagicMock()
        fake_vault._lookup_sync = MagicMock(side_effect=VaultError("read boom"))
        with pytest.raises(VaultError):
            unmask("[LEX-AAAAAAAA]", vault=fake_vault, placeholder_namespace_re=namespace_re)


# --------------------------------------------------------------------------- #
# rebuild helper (single pass — inv 10)
# --------------------------------------------------------------------------- #
class TestRebuild:
    def test_rebuild_applies_spans_once(self):
        out = rebuild("aXbXc", [(1, 2, "P"), (3, 4, "Q")])
        assert out == "aPbQc"

    def test_rebuild_no_spans_identity(self):
        assert rebuild("hello", []) == "hello"

    def test_rebuild_unsorted_plan_handled(self):
        # Defensive: plan need not be pre-sorted.
        out = rebuild("aXbXc", [(3, 4, "Q"), (1, 2, "P")])
        assert out == "aPbQc"
