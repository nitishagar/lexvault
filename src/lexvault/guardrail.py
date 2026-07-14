"""LiteLLM ``CustomGuardrail`` integration for lexvault.

``LexVaultGuardrail`` overrides ONLY the four individual hooks — it deliberately
does NOT define ``apply_guardrail`` (IMPLICIT_SPEC invariant 4): defining it
would reroute dispatch through the unified translation path and reintroduce the
RC1 tool-key-contamination 400 error (``proxy/utils.py:868``). An
``assert "apply_guardrail" not in LexVaultGuardrail.__dict__`` test guards this.

The vault is the cross-hook state store (invariant 3): ``async_pre_call_hook``
writes mask↔original mappings; both restore hooks look them up by placeholder
string. No mappings live in instance attributes (that would race under the
``asyncio.gather`` moderation of concurrent requests — the Presidio bug).

Per-request config (invariant 24) is read defensively from BOTH ``data["metadata"]``
and ``data["litellm_metadata"]`` (the variable name is route-dependent), then
``["requester_metadata"]["lexvault_*"]`` / ``["headers"]["x-lexvault-*"]``.
``fail_open: false`` by default → mask/restore errors block rather than leak
(invariant 11).
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import litellm.integrations.custom_guardrail as _lcg
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.types.guardrails import GuardrailEventHooks

from lexvault.adapters import anthropic as anthropic_adapter
from lexvault.adapters import openai as openai_adapter
from lexvault.adapters import responses as responses_adapter
from lexvault.config import LexVaultConfig, load_dictionary
from lexvault.detector import Detector
from lexvault.engine import mask as engine_mask
from lexvault.engine import unmask as engine_unmask
from lexvault.logging import remask_logging_payload
from lexvault.streaming.restore import (
    anthropic_error_event,
    openai_error_chunk,
    streaming_restore,
)
from lexvault.vault import MappingVault, VaultError

# ModifyResponseException moved between litellm versions: it's in
# `litellm.exceptions` in recent releases but only available via
# `litellm.integrations.custom_guardrail` in the 1.80.x floor. Resolve at
# runtime via the integration module (which re-exports it everywhere) so we
# work across the full supported range (invariant 22).
ModifyResponseException = _lcg.ModifyResponseException  # type: ignore[attr-defined]

__all__ = ["LexVaultGuardrail", "ModifyResponseException"]

logger = logging.getLogger("lexvault")

# Call types where there is no text content to mask (invariant 21): the guardrail
# no-ops cleanly (skip + debug log) rather than touching embeddings/images/etc.
_NON_TEXT_CALL_TYPES = frozenset(
    {
        "embedding",
        "aembedding",
        "image_generation",
        "aimage_generation",
        "image_edit",
        "aimage_edit",
        "moderation",
        "amoderation",
        "atranscription",
        "transcription",
        "aspeech",
        "speech",
        "rerank",
        "arerank",
    }
)

# Call types that use the Anthropic-native request shape.
_ANTHROPIC_CALL_TYPES = frozenset({"anthropic_messages"})
# Call types for the Responses API.
_RESPONSES_CALL_TYPES = frozenset({"responses", "aresponses"})

# Known one-way MASK guardrails that, if listed BEFORE lexvault, can destroy a
# term lexvault intended to reversibly map (invariant 18). Matched by substring
# on the guardrail's class name / registered name.
_ONE_WAY_MASKERS = (
    "content_filter",  # litellm_content_filter with action: MASK
    "presidio",
    "google_text_moderation",
    "llamaguard",
    "aim",
)


class LexVaultGuardrail(CustomGuardrail):
    """Reversible proprietary-term pseudonymization guardrail for LiteLLM.

    Constructed by LiteLLM's file-mount loader with ``litellm_params`` minus
    ``{guardrail, mode, default_on}`` as kwargs. Unknown kwargs are accepted
    (``extra="allow"`` on ``LexVaultConfig`` mirrors the loader's forwarding).
    """

    def __init__(self, guardrail_name: str = "lexvault", **kwargs: Any) -> None:
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.logging_only,
            ],
        )
        self.guardrail_name = guardrail_name
        self._config = self._build_config(kwargs)
        self._detector = self._build_detector(self._config)
        # CS-E7/C6 (invariant 18): one-time guardrail-ordering warning, checked
        # lazily on the first pre_call (the guardrail can't see co-configured
        # guardrails at __init__ — litellm.callbacks is populated after all
        # guardrails register). Warns once if a one-way MASK guardrail precedes
        # lexvault in the callback list, which can defeat reversibility.
        self._ordering_warned = False
        self._vault = MappingVault(self._config.vault_path)
        # Pre-compute the namespace regex + max placeholder length for streaming.
        self._namespace_re = self._config.placeholder_namespace_pattern()
        self._namespace_re_compiled = re.compile(self._namespace_re)
        self._max_placeholder_len = self._compute_max_placeholder_len()

    # NOTE: this class deliberately does NOT define `apply_guardrail`. See
    # IMPLICIT_SPEC invariant 4 and the module docstring. The dispatch check
    # `"apply_guardrail" in type(callback).__dict__` (proxy/utils.py:868) would
    # otherwise reroute through the guardrail_translation handler and reintroduce
    # the RC1 tool-key-contamination 400.

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_config(kwargs: dict[str, Any]) -> LexVaultConfig:
        cfg_kwargs = dict(kwargs)
        # Resolve os.environ/VALUE placeholders for secrets/paths (LiteLLM convention).
        for key in ("org_key", "dictionary_path", "vault_path"):
            if key in cfg_kwargs and isinstance(cfg_kwargs[key], str):
                cfg_kwargs[key] = _resolve_env(cfg_kwargs[key])
        # Load the dictionary file and fold regex terms in.
        terms: list = []
        regex_terms: list = []
        dictionary_path = cfg_kwargs.get("dictionary_path")
        if dictionary_path:
            terms, file_regex = load_dictionary(dictionary_path)
            regex_terms = list(cfg_kwargs.get("regex_terms", [])) + file_regex
        cfg_kwargs["dictionary"] = terms
        cfg_kwargs.setdefault("regex_terms", regex_terms)
        return LexVaultConfig(**cfg_kwargs)

    @staticmethod
    def _build_detector(config: LexVaultConfig) -> Detector:
        return Detector(
            dictionary=config.dictionary,
            regex_terms=config.regex_terms,
            placeholder_namespace=config.placeholder_namespace_pattern(),
        )

    def _compute_max_placeholder_len(self) -> int:
        """Bound for the streaming placeholder window (longest possible placeholder).

        ``[LEX-XXXXXXXX]`` = len(format without {code}) + 8 base32 chars, plus the
        longest ``-N`` suffix we'd realistically emit. We use a generous bound so
        the buffer never splits a placeholder.
        """
        fmt = self._config.placeholder_format
        static_len = len(fmt.replace("{code}", ""))
        code_len = 8
        # Collision suffix up to -9999 (collision loop caps at 1000).
        suffix_len = len("-9999")
        return static_len + code_len + suffix_len

    def _warn_on_guardrail_ordering(self) -> None:
        """CS-E7/C6 (invariant 18): warn once if a one-way MASK guardrail that
        could destroy a masked term is registered BEFORE lexvault.

        LiteLLM runs guardrails in config-list order; a one-way masker preceding
        lexvault can obscure a dictionary term before lexvault maps it, defeating
        reversibility. The guardrail can't see its siblings at ``__init__`` time
        (``litellm.callbacks`` is populated after all guardrails register), so we
        check lazily on the first ``pre_call``. Best-effort: never blocks.
        """
        try:
            import litellm

            callbacks = getattr(litellm, "callbacks", None)
            if not isinstance(callbacks, list):
                return
            self_index = None
            for idx, cb in enumerate(callbacks):
                if cb is self:
                    self_index = idx
                    break
            if self_index is None or self_index == 0:
                return  # lexvault is first (or not yet registered) — fine.
            preceding = callbacks[:self_index]
            names = [
                str(getattr(cb, "guardrail_name", "") or type(cb).__name__).lower()
                for cb in preceding
            ]
            risky = [n for n in names if any(m in n for m in _ONE_WAY_MASKERS)]
            if risky:
                logger.warning(
                    "lexvault: a one-way MASK guardrail (%s) runs before lexvault. "
                    "List lexvault FIRST in your guardrails config so dictionary "
                    "terms are reversibly masked before any one-way masker can "
                    "destroy them (invariant 18).",
                    ", ".join(sorted(set(risky))),
                )
        except Exception:  # noqa: BLE001 - ordering check must never break the call
            return

    # ------------------------------------------------------------------ #
    # pre_call — MASK (invariant 11: fail-closed by default)
    # ------------------------------------------------------------------ #
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        if call_type in _NON_TEXT_CALL_TYPES:
            logger.debug("lexvault: no-op for non-text call_type=%s", call_type)
            return None

        if not isinstance(data, dict):
            return None

        # CS-E7/C6 (invariant 18): one-time guardrail-ordering warning.
        if not self._ordering_warned:
            self._ordering_warned = True
            self._warn_on_guardrail_ordering()

        req_id = data.get("litellm_call_id") or str(uuid.uuid4())
        per_req = _per_request_config(data)
        org_key = self._config.org_key.get_secret_value()
        scope = per_req.get("scope") or self._config.scope
        placeholder_format = self._config.placeholder_format
        mask_system = self._config.mask_system_role

        slots = _request_slots_for_call_type(data, call_type, mask_system=mask_system)
        try:
            for getter, setter in slots:
                text = getter()
                if text:
                    masked = await engine_mask(
                        text,
                        detector=self._detector,
                        vault=self._vault,
                        org_key=org_key,
                        scope=scope,
                        placeholder_format=placeholder_format,
                        request_id=req_id,
                    )
                    setter(masked)
        except (VaultError, Exception) as exc:
            if self._config.fail_open:
                logger.warning("lexvault: fail-OPEN on mask error (fail_open=true): %s", exc)
                return data
            # Fail closed: block the call so no original reaches the upstream LLM.
            logger.error("lexvault: blocking request (mask error, fail_open=false): %s", exc)
            block_exc = _make_block_exception("lexvault unavailable: masking failed")
            raise block_exc from exc

        return data

    # ------------------------------------------------------------------ #
    # post_call_success — RESTORE non-streaming (invariant 1, 11)
    # ------------------------------------------------------------------ #
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        if response is None:
            return response
        slots: list = []
        for slots_fn in (
            openai_adapter.response_slots,
            anthropic_adapter.response_slots,
            responses_adapter.response_slots,
        ):
            try:
                slots.extend(slots_fn(response))
            except Exception:  # noqa: BLE001 - adapter mismatch is non-fatal
                continue
        if not slots:
            return response

        try:
            residual_re = self._namespace_re_compiled
            leaked = False
            for getter, setter in slots:
                text = getter()
                if text:
                    restored = engine_unmask(
                        text, vault=self._vault, placeholder_namespace_re=self._namespace_re
                    )
                    # Fail-closed: an unresolved namespace placeholder in the
                    # response is a potential leak (invariant 11). We re-check
                    # AFTER restore — any remaining placeholder that matches the
                    # namespace but has no mapping is treated as a leak. (This is
                    # distinct from invariant 17's engine-level behavior; on the
                    # response path, an unresolved placeholder is suspicious.)
                    if not self._config.fail_open and residual_re.search(restored):
                        leaked = True
                    setter(restored)
            if leaked:
                logger.error("lexvault: unresolved placeholder in response (fail_open=false)")
                raise ModifyResponseException(
                    message="response filtered: lexvault restore failed",
                    model=_safe_model(data, response),
                    request_data=data if isinstance(data, dict) else {},
                    guardrail_name=self.guardrail_name,
                )
        except ModifyResponseException:
            raise
        except (VaultError, Exception) as exc:
            if self._config.fail_open:
                logger.warning("lexvault: fail-OPEN on restore error: %s", exc)
                return response
            # Fail closed: return a sanitized response with NO placeholder and NO original.
            logger.error("lexvault: sanitizing response (restore error, fail_open=false): %s", exc)
            raise ModifyResponseException(
                message="response filtered: lexvault restore failed",
                model=_safe_model(data, response),
                request_data=data if isinstance(data, dict) else {},
                guardrail_name=self.guardrail_name,
            ) from exc

        return response

    # ------------------------------------------------------------------ #
    # post_call_streaming_iterator — RESTORE streaming (invariant 1, 5, 6, 11)
    # ------------------------------------------------------------------ #
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[Any, None]:
        if response is None:
            return
            yield  # pragma: no cover - make this an async generator for typing
        try:
            async for chunk in streaming_restore(
                response,
                vault=self._vault,
                placeholder_namespace_re=self._namespace_re,
                max_placeholder_len=self._max_placeholder_len,
            ):
                yield chunk
        except Exception as exc:
            # Fail closed: emit one error frame, then stop. Never yield a partial
            # placeholder. StreamingError covers dangling-partial + restore-flush
            # failures; a VaultError (or any unexpected error) mid-loop is routed
            # through the same sanitized error frame (CS-E5) rather than aborting
            # the stream with a raw traceback.
            # Security: the client-facing frame carries a FIXED generic message —
            # never str(exc), which could embed a placeholder/original if a future
            # exception echoes request content. The detail is logged server-side.
            logger.error("lexvault: closing stream after restore error: %s", exc)
            safe_msg = "lexvault: stream restore failed"
            # The Anthropic /v1/messages route yields raw SSE bytes; emit an
            # Anthropic error event there, else an OpenAI-style error chunk.
            if _is_anthropic_messages_route(request_data):
                yield anthropic_error_event(safe_msg)
            else:
                yield openai_error_chunk(safe_msg)

    # ------------------------------------------------------------------ #
    # logging — defense-in-depth re-mask (invariant 19)
    # ------------------------------------------------------------------ #
    async def async_logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> tuple[dict, Any]:
        try:
            return await remask_logging_payload(
                kwargs,
                result,
                call_type=call_type,
                detector=self._detector,
                vault=self._vault,
                org_key=self._config.org_key.get_secret_value(),
                scope=self._config.scope,
                placeholder_format=self._config.placeholder_format,
                mask_system=self._config.mask_system_role,
                request_id=(kwargs.get("litellm_call_id") if isinstance(kwargs, dict) else None),
            )
        except Exception as exc:  # noqa: BLE001 - logging must never crash the call
            logger.warning("lexvault: logging re-mask skipped (non-fatal): %s", exc)
            return kwargs, result


# --------------------------------------------------------------------------- #
# per-request config (invariant 24) + request-slot dispatch
# --------------------------------------------------------------------------- #
def _per_request_config(data: dict) -> dict[str, str]:
    """Read per-request config defensively from BOTH metadata locations.

    The metadata variable name is ROUTE-DEPENDENT (litellm_pre_call_utils.py:71-85):
    ``litellm_metadata`` on /v1/messages, /v1/responses, batches, files;
    ``metadata`` on /v1/chat/completions and everything else. Research F5: read
    both defensively. Then ``["requester_metadata"]["lexvault_*"]`` and
    ``["headers"]["x-lexvault-*"]``.
    """
    out: dict[str, str] = {}
    for meta_key in ("metadata", "litellm_metadata"):
        meta = data.get(meta_key)
        if not isinstance(meta, dict):
            continue
        requester = meta.get("requester_metadata")
        if isinstance(requester, dict):
            for k, v in requester.items():
                if isinstance(k, str) and k.startswith("lexvault_"):
                    out[k.removeprefix("lexvault_")] = v
        headers = meta.get("headers")
        if isinstance(headers, dict):
            for k, v in headers.items():
                if isinstance(k, str) and k.lower().startswith("x-lexvault-"):
                    out[k.lower().removeprefix("x-lexvault-")] = v
    return out


def _request_slots_for_call_type(data: dict, call_type: str, *, mask_system: bool) -> list:
    if call_type in _RESPONSES_CALL_TYPES:
        return responses_adapter.request_input_slots(data)
    if call_type in _ANTHROPIC_CALL_TYPES:
        return anthropic_adapter.request_message_slots(data, mask_system=mask_system)
    return openai_adapter.request_message_slots(data, mask_system=mask_system)


def _safe_model(data: Any, response: Any) -> str:
    if isinstance(data, dict):
        model = data.get("model")
        if isinstance(model, str):
            return model
    m = getattr(response, "model", None) if response is not None else None
    return str(m) if m else "unknown"


def _is_anthropic_messages_route(request_data: Any) -> bool:
    """Heuristic: is this a /v1/messages (native Anthropic) streaming response?

    The native Anthropic route yields raw SSE bytes; the OpenAI / Responses
    routes yield parsed objects. We infer from ``request_data``'s call-type /
    route hints when present. Defaults to False (OpenAI-shape error) which is the
    safe majority case; the streaming dispatcher handles both chunk types anyway.
    """
    if not isinstance(request_data, dict):
        return False
    call_type = request_data.get("call_type") or request_data.get("route_type")
    if call_type == "anthropic_messages":
        return True
    # LiteLLM tags the route in proxy_server_request.
    proxy_req = request_data.get("proxy_server_request")
    if isinstance(proxy_req, dict):
        url = proxy_req.get("url", "")
        if isinstance(url, str) and "/v1/messages" in url:
            return True
    return False


def _resolve_env(value: str) -> str:
    """Resolve ``os.environ/NAME`` style placeholders (LiteLLM convention)."""
    if isinstance(value, str) and value.startswith("os.environ/"):
        env_name = value.removeprefix("os.environ/")
        resolved = os.environ.get(env_name)
        return resolved if resolved is not None else value
    return value


def _make_block_exception(detail: str) -> Exception:
    """Build an exception that LiteLLM's proxy will surface as an HTTP 503.

    ``fastapi`` is an optional litellm proxy dependency (present when running
    under ``litellm --config``, absent in library-only use). We import it lazily
    and fall back to a plain ``RuntimeError`` so the guardrail still blocks
    (raises) outside a proxy context.
    """
    try:
        from fastapi import HTTPException

        exc: Exception = HTTPException(status_code=503, detail=detail)
        return exc
    except ImportError:
        return RuntimeError(detail)
