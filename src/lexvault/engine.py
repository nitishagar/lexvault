"""Engine — placeholder derivation + mask/unmask.

``derive_placeholder`` produces a deterministic, collision-resistant code from
(term, org_key, scope) via HMAC-SHA256 → base32[:8]. Determinism gives
cross-surface/cross-turn consistency (IMPLICIT_SPEC invariants 2, 9); the vault
appends a ``-N`` suffix on the (rare) truncated-HMAC collision.

``mask`` runs the detector, derives a placeholder for each match, records it in
the vault, and rebuilds the string **once** left-to-right (invariant 10 — NOT a
``str.replace`` loop). ``unmask`` finds namespace-regex spans and replaces only
those (invariant 17 — user-typed lookalikes are left alone unless they're a
known mapping).

The engine is pure w.r.t. text transformation; it takes the vault as a
dependency for collision resolution and restore lookups.
"""

from __future__ import annotations

import base64
import hmac
import re
from collections.abc import Callable

from lexvault.detector import Detector, Match
from lexvault.vault import MappingVault

__all__ = ["derive_placeholder", "mask", "unmask", "mask_then_unmask"]


def derive_placeholder(
    term: str, org_key: str, scope: str, placeholder_format: str = "[LEX-{code}]"
) -> str:
    """Derive a deterministic placeholder for ``term`` under (org_key, scope).

    HMAC-SHA256(org_key, ``scope\\x1fterm``) → first 5 bytes → base32 → first 8
    chars. The scope and term are separated by a unit-separator (``\\x1f``) so a
    scope/term boundary can't be ambiguous. The result is formatted with
    ``placeholder_format`` (which must contain ``{code}``).
    """
    digest = hmac.new(org_key.encode(), f"{scope}\x1f{term}".encode(), "sha256").digest()
    code = base64.b32encode(digest[:5]).decode("ascii")[:8]  # 5 bytes → 8 base32 chars
    return placeholder_format.format(code=code)


async def mask(
    text: str,
    *,
    detector: Detector,
    vault: MappingVault,
    org_key: str,
    scope: str,
    placeholder_format: str,
    request_id: str | None,
) -> str:
    """Mask dictionary/regex terms in ``text`` → text with placeholders.

    Single rebuild from detector spans (invariant 10). Each placeholder is
    recorded in the vault (collision-resolved). Already-present placeholders are
    pre-excluded by the detector (idempotency, invariant 16). Fail-closed: a
    vault error propagates (invariant 11).
    """
    if not text:
        return text

    matches = detector.find_matches(text)
    if not matches:
        return text

    # Build the replacement plan as (start, end, placeholder) and track which
    # originals need vault assignment. We assign-via-vault so collisions get a
    # deterministic suffix and restore always works.
    plan: list[tuple[int, int, str]] = []
    for m in matches:
        placeholder = derive_placeholder(m.term, org_key, scope, placeholder_format)
        final = await vault.assign(
            scope, placeholder, _slice_match(text, m), request_id=request_id, term_type=m.type
        )
        plan.append((m.start, m.end, final))

    return _rebuild(text, plan)


def unmask(
    text: str,
    *,
    vault: MappingVault,
    placeholder_namespace_re: str,
) -> str:
    """Replace placeholder-namespace spans in ``text`` with their originals.

    Only spans matching ``placeholder_namespace_re`` are considered, and only
    those that resolve to a stored mapping are replaced (invariant 17: a
    user-typed lookalike with no mapping is left untouched). Synchronous — uses
    the vault's lock-free WAL read. Fails closed: a vault error propagates.
    """
    if not text:
        return text

    rx = re.compile(placeholder_namespace_re)
    plan: list[tuple[int, int, str]] = []
    for m in rx.finditer(text):
        original = vault._lookup_sync(m.group(0))  # noqa: SLF001 — sync read, lock-free under WAL
        if original is not None:
            plan.append((m.start(), m.end(), original))
    if not plan:
        return text
    return _rebuild(text, plan)


async def mask_then_unmask(
    text: str,
    *,
    detector: Detector,
    vault: MappingVault,
    org_key: str,
    scope: str,
    placeholder_format: str,
    placeholder_namespace_re: str,
    request_id: str | None,
) -> str:
    """Convenience: mask then immediately unmask. Identity on in-dictionary text."""
    masked = await mask(
        text,
        detector=detector,
        vault=vault,
        org_key=org_key,
        scope=scope,
        placeholder_format=placeholder_format,
        request_id=request_id,
    )
    return unmask(masked, vault=vault, placeholder_namespace_re=placeholder_namespace_re)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _slice_match(text: str, m: Match) -> str:
    return text[m.start : m.end]


def _rebuild(text: str, plan: list[tuple[int, int, str]]) -> str:
    """Rebuild ``text`` applying ``(start, end, replacement)`` spans once.

    Assumes ``plan`` spans are non-overlapping and sorted-able; we sort by start
    defensively. Non-overlap is guaranteed by the detector's overlap resolution
    for ``mask`` and by regex finditer for ``unmask``.
    """
    if not plan:
        return text
    plan_sorted = sorted(plan, key=lambda p: p[0])
    out: list[str] = []
    cursor = 0
    for start, end, replacement in plan_sorted:
        out.append(text[cursor:start])
        out.append(replacement)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


# Exposed for unit tests that want the pure rebuild helper directly.
rebuild: Callable[[str, list[tuple[int, int, str]]], str] = _rebuild
