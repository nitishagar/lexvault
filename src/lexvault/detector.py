"""Detector — dictionary + regex span finder.

Builds Aho-Corasick automata from dictionary terms ONCE at construction
(O(Σ pattern lengths) build, O(text length) matching, no backtracking) and
compiles regex terms once. ``find_matches`` collects ALL match spans, merges
dictionary + regex spans, resolves overlaps **longest-then-leftmost**
deterministically, and pre-excludes spans that fall inside a placeholder
namespace (idempotency, IMPLICIT_SPEC invariant 16). Pure and stateless — no
I/O, no vault access.

The single-pass overlap resolution + single rebuild is what makes masking
iteration-order independent (invariant 10) — this is NOT a ``str.replace`` loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering

import ahocorasick

from lexvault.config import DictionaryTerm, RegexTerm

__all__ = ["Match", "MatchSource", "Detector"]


class MatchSource(str, Enum):
    """Where a match came from — dictionary (Aho-Corasick) or regex."""

    DICTIONARY = "dictionary"
    REGEX = "regex"


@dataclass(frozen=True)
class Match:
    """A detected span ``text[start:end]``.

    ``term`` is the matched literal for dictionary matches, or the regex
    ``name`` for regex matches. ``source`` distinguishes the two.
    """

    start: int
    end: int
    term: str
    type: str
    source: MatchSource


@total_ordering
class _Span:
    """A candidate span with deterministic overlap ordering.

    Ordering: longer span first (so it wins overlaps); on a length tie, smaller
    ``start`` first (leftmost wins). ``total_ordering`` supplies the rest.
    """

    __slots__ = ("start", "end", "term", "type", "source")

    def __init__(self, start: int, end: int, match: Match) -> None:
        self.start = start
        self.end = end
        self.term = match.term
        self.type = match.type
        self.source = match.source

    @property
    def _sort_key(self) -> tuple[int, int]:
        # Longer length first → negate; then leftmost start first.
        return (-(self.end - self.start), self.start)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Span):
            return NotImplemented
        return self._sort_key == other._sort_key

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _Span):
            return NotImplemented
        return self._sort_key < other._sort_key

    def __hash__(self) -> int:  # pragma: no cover - spans aren't hashed in practice
        return hash(self._sort_key)

    def overlaps(self, other: _Span) -> bool:
        return self.start < other.end and other.start < self.end

    def to_match(self) -> Match:
        return Match(self.start, self.end, self.term, self.type, self.source)


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and b0 < a1


class Detector:
    """Detect dictionary + regex spans in text.

    Constructed ONCE per guardrail (the Aho-Corasick automata and compiled
    regexes are reused across requests). ``placeholder_namespace`` is the
    regex string used to pre-exclude already-present placeholders.

    Case sensitivity: case-insensitive terms are collected in a lowercased
    automaton and matched against a lowercased copy of the text; case-sensitive
    terms are collected in an exact automaton and matched against the original
    text. ``str.lower`` is length-preserving for ASCII (the realistic domain for
    enterprise dictionary terms); Unicode case-folding that changes length is
    out of scope for v0.1 and documented as such.
    """

    def __init__(
        self,
        dictionary: list[DictionaryTerm],
        regex_terms: list[RegexTerm],
        placeholder_namespace: str,
    ) -> None:
        self._placeholder_re = re.compile(placeholder_namespace)
        self._compiled_regexes: list[tuple[RegexTerm, re.Pattern[str]]] = []
        self._ci_automaton = ahocorasick.Automaton()
        self._cs_automaton = ahocorasick.Automaton()
        self._build_automata(dictionary)
        self._compile_regexes(regex_terms)

    def _build_automata(self, dictionary: list[DictionaryTerm]) -> None:
        """Build two automata: case-insensitive (lowercased) and case-sensitive (exact)."""
        # Deduplicate by key within each automaton.
        ci_seen: dict[str, tuple[str, str]] = {}
        cs_seen: dict[str, tuple[str, str]] = {}
        for term in dictionary:
            if term.case_sensitive:
                cs_seen.setdefault(term.term, (term.term, term.type))
            else:
                ci_seen.setdefault(term.term.lower(), (term.term, term.type))
        for key, (literal, ttype) in ci_seen.items():
            self._ci_automaton.add_word(key, (literal, ttype))
        for key, (literal, ttype) in cs_seen.items():
            self._cs_automaton.add_word(key, (literal, ttype))
        self._ci_automaton.make_automaton()
        self._cs_automaton.make_automaton()

    def _compile_regexes(self, regex_terms: list[RegexTerm]) -> None:
        for rt in regex_terms:
            flags = 0 if rt.case_sensitive else re.IGNORECASE
            self._compiled_regexes.append((rt, re.compile(rt.pattern, flags)))

    def find_matches(self, text: str) -> list[Match]:
        """Find all non-overlapping dictionary + regex spans in ``text``.

        Overlaps are resolved longest-then-leftmost. Spans that overlap a
        placeholder-namespace region are pre-excluded (invariant 16: an
        already-present placeholder in conversation history is left untouched).
        Returns matches sorted by start position.
        """
        if not text:
            return []

        candidates = self._collect_candidates(text)
        if not candidates:
            return []

        # Pre-exclude spans overlapping any placeholder-namespace region.
        excluded = self._placeholder_regions(text)
        if excluded:
            candidates = [
                s
                for s in candidates
                if not any(_ranges_overlap(s.start, s.end, e0, e1) for e0, e1 in excluded)
            ]
            if not candidates:
                return []

        # Resolve overlaps: sort longest-then-leftmost, greedily keep spans that
        # don't overlap an already-kept span.
        candidates.sort()
        kept: list[_Span] = []
        for span in candidates:
            if any(span.overlaps(k) for k in kept):
                continue
            kept.append(span)

        kept.sort(key=lambda s: s.start)
        return [s.to_match() for s in kept]

    def _collect_candidates(self, text: str) -> list[_Span]:
        """Collect ALL dictionary + regex spans (before overlap resolution)."""
        spans: list[_Span] = []
        self._add_dictionary_spans(text, spans)
        for rt, rx in self._compiled_regexes:
            for m in rx.finditer(text):
                spans.append(
                    _Span(
                        m.start(),
                        m.end(),
                        Match(m.start(), m.end(), rt.name, rt.type, MatchSource.REGEX),
                    )
                )
        return spans

    def _add_dictionary_spans(self, text: str, out: list[_Span]) -> None:
        """Append Aho-Corasick dictionary spans to ``out``.

        Case-insensitive terms match against ``text.lower()``; case-sensitive
        terms match against ``text`` exactly. ``str.lower`` is length-preserving
        for ASCII so offsets are identical to the original text.
        """
        # Case-insensitive automaton (lowercased keys → lowercased search text).
        if len(self._ci_automaton) > 0:
            lowered = text.lower()
            for end_idx, (term, ttype) in self._ci_automaton.iter(lowered):
                # Re-validate: pyahocorasick yields the longest match ending at
                # end_idx; we still confirm the slice matches under case rules.
                start = end_idx - len(term.lower()) + 1
                out.append(
                    _Span(
                        start,
                        end_idx + 1,
                        Match(start, end_idx + 1, term, ttype, MatchSource.DICTIONARY),
                    )
                )
        # Case-sensitive automaton (exact keys → original search text).
        if len(self._cs_automaton) > 0:
            for end_idx, (term, ttype) in self._cs_automaton.iter(text):
                start = end_idx - len(term) + 1
                out.append(
                    _Span(
                        start,
                        end_idx + 1,
                        Match(start, end_idx + 1, term, ttype, MatchSource.DICTIONARY),
                    )
                )

    def _placeholder_regions(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) regions of text matching the placeholder namespace."""
        return [(m.start(), m.end()) for m in self._placeholder_re.finditer(text)]
