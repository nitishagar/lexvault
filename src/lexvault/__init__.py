"""lexvault — reversible proprietary-term pseudonymization for LiteLLM.

A LiteLLM ``CustomGuardrail`` plugin that masks an enterprise's own dictionary
terms (codenames, products, customers, schema) to deterministic placeholders on
the request path and faithfully restores them on the response path — across
non-streaming OpenAI + Anthropic-native, streaming (``ModelResponseStream``
*and* raw Anthropic SSE bytes), and tool-call arguments.

Public surface:
    LexVaultGuardrail — the LiteLLM guardrail class (mounted via the shim).
    LexVaultConfig    — pydantic config model for the guardrail.
    MappingVault      — the local SQLite mapping store.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Re-exported public names. These are (re)defined in their real modules
# (config.py / vault.py / guardrail.py) as the phases land; Phase 1 ships a
# working placeholder so `import lexvault` and the shim are importable now.
from lexvault.config import LexVaultConfig
from lexvault.guardrail import LexVaultGuardrail
from lexvault.vault import MappingVault

__all__ = [
    "LexVaultGuardrail",
    "LexVaultConfig",
    "MappingVault",
    "__version__",
]
