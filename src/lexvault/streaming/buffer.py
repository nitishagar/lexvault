"""Restore-aware bounded buffer for streaming.

This is the crux of correct streaming restore (IMPLICIT_SPEC invariant 5 + the
RC4 split-placeholder fix the MVP exists to provide). The naive approach — buffer
the trailing window, then call ``unmask`` on each emitted slice — LEAKS when a
placeholder straddles the window cut: no single ``unmask`` call sees the whole
placeholder, so it passes through unrestored.

The correct approach implemented here: accumulate the FULL text stream, restore
complete placeholders, and only emit text up to the point where we are certain no
partial placeholder is building. A partial placeholder is signaled by an
unmatched opener (e.g. ``[LEX-``) trailing into the held window. We hold back at
most ``max_placeholder_len`` trailing chars — the longest possible placeholder —
so the emitted text never contains a half-placeholder that ``unmask`` could miss.
"""

from __future__ import annotations

import re
from collections.abc import Callable

__all__ = ["PlaceholderBuffer"]

# A placeholder→original lookup: takes the placeholder string, returns the
# original or None if no mapping exists.
Lookup = Callable[[str], "str | None"]


class PlaceholderBuffer:
    """Accumulate text, restore placeholders, emit only text that has provably
    left any placeholder boundary.

    Usage (streaming restore)::

        buf = PlaceholderBuffer(window, namespace_re)
        for chunk in chunks:
            buf.feed(chunk_text)
            yield buf.drain_restored_with_vault(vault._lookup_sync)  # safe text
        tail, partial = buf.flush_restored_with_vault(vault._lookup_sync)
        if partial:  # dangling partial → fail closed
            ...
    """

    def __init__(self, max_placeholder_len: int, placeholder_namespace_re: str) -> None:
        if max_placeholder_len <= 0:
            msg = "max_placeholder_len must be positive"
            raise ValueError(msg)
        self._window = max_placeholder_len
        self._ns_re = re.compile(placeholder_namespace_re)
        self._opener = _leading_literal(placeholder_namespace_re)
        self._buf = ""

    def feed(self, text: str) -> None:
        """Append text to the accumulated stream."""
        if text:
            self._buf += text

    def drain_restored_with_vault(self, lookup: Lookup) -> str:
        """Emit text that has provably left any placeholder boundary, restored.

        We hold back the trailing ``<= window`` chars PLUS any partial
        placeholder opener that straddles the window cut (an opener in the ready
        portion whose closing ``]`` would fall in the held window). This is the
        RC4 correctness core: a placeholder is never split across an emit, so
        ``_restore_inplace`` always sees complete placeholders.

        We ALSO hold back a trailing *proper prefix* of the opener (e.g. ``[``,
        ``[L``, ``[LE``, ``[LEX``) at the end of the ready slice — if the held
        window completes it into a full placeholder, emitting the prefix now
        would split the placeholder across an emit (CS-E1: the straddle leak).
        """
        if len(self._buf) <= self._window:
            return ""
        cut = len(self._buf) - self._window
        ready_candidate = self._buf[:cut]
        # Hold back from the leftmost detected partial-opener boundary:
        #  (a) a full unclosed opener anywhere in the ready slice, or
        #  (b) a trailing proper prefix of the opener at the slice's end.
        # Both signal the start of a placeholder that may complete in the held
        # window; holding keeps that placeholder whole so _restore_inplace sees
        # it in one piece.
        hold = self._hold_back_point(ready_candidate)
        if hold >= 0:
            cut = hold
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return _restore_inplace(ready, self._ns_re, lookup)

    def flush(self) -> tuple[str, bool]:
        """Drain the buffer on stream end.

        Returns ``(remaining, partial_in_namespace)``. ``partial_in_namespace`` is
        True if the held tail contains an unclosed placeholder opener (the caller
        should fail-closed / sanitize rather than emit a partial that could be a
        real leaked placeholder).
        """
        remaining = self._buf
        self._buf = ""
        partial_in_namespace = self._has_unclosed_opener(remaining)
        return remaining, partial_in_namespace

    def flush_restored_with_vault(self, lookup: Lookup) -> tuple[str, bool]:
        """Like :meth:`flush` but restores complete placeholders in the tail."""
        remaining = self._buf
        self._buf = ""
        partial_in_namespace = self._has_unclosed_opener(remaining)
        restored = _restore_inplace(remaining, self._ns_re, lookup)
        return restored, partial_in_namespace

    @property
    def held_bytes(self) -> int:
        return len(self._buf)

    def _hold_back_point(self, ready: str) -> int:
        """Index in ``ready`` at/before which text is safe to emit, else -1.

        Returns the leftmost position we must hold back from, considering:
          (a) a full unclosed opener in ``ready`` (its close would fall in the
              held window), and
          (b) a trailing proper prefix of the opener at the end of ``ready``
              (``[``, ``[L``, ``[LE``, ``[LEX``) — if the held window completes
              it into a full placeholder, emitting the prefix now would split
              the placeholder (CS-E1 straddle leak).

        Holding from the smaller of the two keeps the whole potential
        placeholder in the buffer until its close arrives or stream ends.
        """
        full = self._last_unclosed_opener(ready)
        prefix = self._trailing_opener_prefix_len(ready)
        candidates = [c for c in (full, prefix) if c >= 0]
        return min(candidates) if candidates else -1

    def _has_unclosed_opener(self, text: str) -> bool:
        """True if ``text`` ends with an unclosed placeholder opener OR a proper
        prefix of the opener (a potential partial placeholder)."""
        return self._last_unclosed_opener(text) >= 0 or self._trailing_opener_prefix_len(text) >= 0

    def _last_unclosed_opener(self, text: str) -> int:
        """Index of the last opener in ``text`` not followed by ``]``, else -1.

        E.g. ``"ending [LEX-AAA"`` → the opener ``[LEX-`` appears and no ``]``
        follows it → its index. A complete placeholder (``[LEX-AAAAAAAA]]``) or
        plain text → -1.
        """
        if not self._opener or not text:
            return -1
        idx = text.rfind(self._opener)
        if idx < 0:
            return -1
        tail = text[idx:]
        if "]" in tail:
            return -1
        return idx

    def _trailing_opener_prefix_len(self, text: str) -> int:
        """Start index of a trailing proper prefix of the opener, else -1.

        CS-E1: if ``text`` ends with a proper (non-empty, non-full) prefix of
        the opener — e.g. ``[``, ``[L``, ``[LE``, ``[LEX`` for opener ``[LEX-`` —
        the held window may complete it into a full placeholder. Returns the
        index where that prefix begins so the caller can hold it back.

        Only the LONGEST trailing prefix matters (it dominates shorter ones):
        for opener ``[LEX-`` on text ending ``...X[LEX``, the prefix ``[LEX``
        (length 4) begins at ``len(text)-4``.
        """
        if not self._opener or not text:
            return -1
        # Longest trailing substring of `text` that is a proper prefix of the
        # opener. Bound the scan by the shorter of the two strings.
        max_len = min(len(self._opener) - 1, len(text))
        for n in range(max_len, 0, -1):
            if text[-n:] == self._opener[:n]:
                return len(text) - n
        return -1


def _restore_inplace(text: str, ns_re: re.Pattern[str], lookup: Lookup) -> str:
    """Replace namespace matches in ``text`` with their looked-up originals.

    ``lookup`` is a callable ``placeholder -> str | None``. Only spans that
    resolve to a stored mapping are replaced; others are left (engine invariant
    17 at the text level; the guardrail decides fail-closed on residuals).
    """
    if not text:
        return text
    out: list[str] = []
    cursor = 0
    found_any = False
    for m in ns_re.finditer(text):
        original = lookup(m.group(0))
        if original is None:
            continue
        found_any = True
        out.append(text[cursor : m.start()])
        out.append(original)
        cursor = m.end()
    if not found_any:
        return text
    out.append(text[cursor:])
    return "".join(out)


def _leading_literal(regex_pattern: str) -> str:
    """Return the leading literal substring of a regex (un-escaping ``\\X``).

    Stops at the first regex metacharacter that isn't an escaped literal.
    """
    out: list[str] = []
    i = 0
    while i < len(regex_pattern):
        c = regex_pattern[i]
        if c == "\\" and i + 1 < len(regex_pattern):
            out.append(regex_pattern[i + 1])
            i += 2
            continue
        if c in r".^$*+?()[]{}|":
            break
        out.append(c)
        i += 1
    return "".join(out)
