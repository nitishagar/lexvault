"""Unit tests for the detector: dictionary + regex span finding + overlap.

Covers IMPLICIT_SPEC invariants:
  9 (longest wins, no corruption), 10 (iteration-order independent), 16
  (idempotency — placeholder spans pre-excluded).
"""

from __future__ import annotations

import re

from lexvault.config import DictionaryTerm, RegexTerm, placeholder_namespace_regex
from lexvault.detector import Detector, MatchSource

NS = placeholder_namespace_regex("[LEX-{code}]")


def _terms(*vals: str) -> list[DictionaryTerm]:
    return [DictionaryTerm(term=v) for v in vals]


class TestBasicDetection:
    def test_finds_single_term(self):
        d = Detector(_terms("Project Titan"), [], NS)
        matches = d.find_matches("review Project Titan now")
        assert len(matches) == 1
        m = matches[0]
        assert m.start == 7
        assert m.end == 20
        assert m.term == "Project Titan"
        assert m.source is MatchSource.DICTIONARY

    def test_finds_multiple_non_overlapping(self):
        d = Detector(_terms("Project Titan", "customer_database"), [], NS)
        matches = d.find_matches("Project Titan uses customer_database")
        assert {m.term for m in matches} == {"Project Titan", "customer_database"}

    def test_case_insensitive_by_default(self):
        d = Detector(_terms("Project Titan"), [], NS)
        matches = d.find_matches("project titan and PROJECT TITAN")
        assert len(matches) == 2

    def test_case_sensitive_when_requested(self):
        d = Detector([DictionaryTerm(term="Project Titan", case_sensitive=True)], [], NS)
        matches = d.find_matches("project titan and Project Titan")
        assert len(matches) == 1
        assert matches[0].term == "Project Titan"

    def test_no_matches_returns_empty(self):
        d = Detector(_terms("Project Titan"), [], NS)
        assert d.find_matches("nothing here") == []

    def test_empty_text_returns_empty(self):
        d = Detector(_terms("Project Titan"), [], NS)
        assert d.find_matches("") == []


class TestRegex:
    def test_regex_match_found(self):
        d = Detector([], [RegexTerm(name="Employee ID", pattern=r"EMP-\d{4,6}")], NS)
        matches = d.find_matches("contact EMP-123456 or EMP-99")
        # EMP-123456 matches the full pattern; EMP-99 (only 2 digits) does not.
        ids = [m for m in matches if m.source is MatchSource.REGEX]
        assert len(ids) == 1
        assert ids[0].term == "Employee ID"
        assert "EMP-123456" in "contact EMP-123456 or EMP-99"[ids[0].start : ids[0].end]

    def test_dictionary_and_regex_coexist(self):
        d = Detector(
            _terms("Project Titan"),
            [RegexTerm(name="Employee ID", pattern=r"EMP-\d{4,6}")],
            NS,
        )
        matches = d.find_matches("Project Titan is EMP-123456")
        assert len(matches) == 2
        assert {m.source for m in matches} == {MatchSource.DICTIONARY, MatchSource.REGEX}


class TestOverlapResolution:
    def test_longest_wins_over_substring(self):
        d = Detector(_terms("Project Titan", "Titan"), [], NS)
        matches = d.find_matches("Project Titan")
        assert len(matches) == 1
        assert matches[0].term == "Project Titan"

    def test_adjacent_distinct_both_found(self):
        d = Detector(_terms("Project Titan", "Titan"), [], NS)
        matches = d.find_matches("Project Titan and Titan")
        assert {m.term for m in matches} == {"Project Titan", "Titan"}

    def test_overlap_with_regex_longer_wins(self):
        # A regex matching "EMP-123456789" (9 digits) vs a dict term "EMP" — the
        # longer regex span should win where they overlap.
        d = Detector(
            _terms("EMP"),
            [RegexTerm(name="LongEmp", pattern=r"EMP-\d{6}")],
            NS,
        )
        matches = d.find_matches("EMP-123456")
        # The regex matches the full "EMP-123456"; dict "EMP" is a prefix that
        # overlaps it → longest wins.
        assert len(matches) == 1
        assert matches[0].end - matches[0].start == len("EMP-123456")


class TestIdempotencyPreExclusion:
    def test_existing_placeholder_not_rematched(self):
        """A placeholder already in text is pre-excluded (invariant 16)."""
        d = Detector(_terms("Project Titan"), [], NS)
        # A real placeholder-lookalike in the text should NOT be detected as a
        # term, and a term overlapping a placeholder span is excluded.
        text = "Prior [LEX-AAAAAAAA]. Also Project Titan."
        matches = d.find_matches(text)
        # "Project Titan" is found; nothing inside the placeholder span.
        assert any(m.term == "Project Titan" for m in matches)
        ns_re = re.compile(NS)
        placeholder_span = ns_re.search(text).span()
        for m in matches:
            # No match overlaps the placeholder.
            assert not (m.start < placeholder_span[1] and placeholder_span[0] < m.end)


class TestIterationOrderIndependence:
    def test_output_independent_of_dict_order(self):
        text = "Project Titan and Titan near customer_database"
        d1 = Detector(_terms("Project Titan", "Titan", "customer_database"), [], NS)
        d2 = Detector(list(reversed(_terms("Project Titan", "Titan", "customer_database"))), [], NS)
        m1 = [(m.start, m.end, m.term) for m in d1.find_matches(text)]
        m2 = [(m.start, m.end, m.term) for m in d2.find_matches(text)]
        assert m1 == m2
