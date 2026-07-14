"""lexvault file-mount shim for LiteLLM's naive ``split('.')`` loader.

LiteLLM's ``initialize_custom_guardrail`` does ``guardrail.split('.')`` and loads
``<file>.py`` from the config dir (``guardrail_registry.py:501,510-512``), so a
``pip install``ed package cannot be referenced directly. Mount this file next to
your ``config.yaml`` and reference it as::

    guardrails:
      - guardrail: lexvault_shim.LexVaultGuardrail
"""

from lexvault import LexVaultGuardrail as LexVaultGuardrail  # noqa: F401
