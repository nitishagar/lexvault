"""LiteLLM ``CustomGuardrail`` integration for lexvault.

Phase 1: placeholder. The real ``LexVaultGuardrail`` (individual hooks only,
NO ``apply_guardrail`` — invariant 4) lands in Phase 3.
"""

from __future__ import annotations


class LexVaultGuardrail:  # noqa: D101 — replaced in Phase 3.
    """Placeholder. Real LiteLLM guardrail lands in Phase 3.

    NOTE: the real class will deliberately NOT define ``apply_guardrail``
    (IMPLICIT_SPEC invariant 4; ``proxy/utils.py:868``). This placeholder is a
    bare object and also defines none of the LiteLLM hooks yet.
    """

    def __init__(self, guardrail_name: str = "lexvault", **_kwargs: object) -> None:
        """Accept arbitrary kwargs so the skeleton imports without a config."""
        self.guardrail_name = guardrail_name
