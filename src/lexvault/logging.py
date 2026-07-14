"""Logging hook helpers — defense-in-depth re-mask (IMPLICIT_SPEC invariant 19).

The original (unmasked) term MUST NOT appear in any log/trace/standard logging
payload. ``async_logging_hook`` re-masks the request payload (already masked by
pre_call, but defense-in-depth) and the response payload before LiteLLM emits
its standard logging (Langfuse/DataDog/OTel). This module provides the pure
re-mask helpers the guardrail calls.
"""

from __future__ import annotations

from typing import Any

from lexvault.adapters import anthropic as anthropic_adapter
from lexvault.adapters import openai as openai_adapter
from lexvault.adapters import responses as responses_adapter
from lexvault.detector import Detector
from lexvault.engine import mask as engine_mask
from lexvault.vault import MappingVault

__all__ = ["remask_logging_payload"]


async def remask_logging_payload(
    kwargs: dict,
    result: Any,
    *,
    call_type: str,
    detector: Detector,
    vault: MappingVault,
    org_key: str,
    scope: str,
    placeholder_format: str,
    mask_system: bool,
    request_id: str | None,
) -> tuple[dict, Any]:
    """Re-mask ``(kwargs, result)`` for the logging hook (invariant 19).

    The request side is already masked by ``async_pre_call_hook``; we re-mask
    defensively in case a logging path re-reads the raw kwargs. The response
    side is re-masked too (defense-in-depth: even if restore already ran,
    logging must never show originals).
    """
    # --- request side (kwargs) ---
    if isinstance(kwargs, dict):
        await _remask_request(
            kwargs,
            call_type,
            detector,
            vault,
            org_key,
            scope,
            placeholder_format,
            mask_system,
            request_id,
        )

    # --- response side (result) ---
    if result is not None:
        await _remask_response(
            result, detector, vault, org_key, scope, placeholder_format, request_id
        )

    return kwargs, result


async def _remask_request(
    data: dict,
    call_type: str,
    detector: Detector,
    vault: MappingVault,
    org_key: str,
    scope: str,
    placeholder_format: str,
    mask_system: bool,
    request_id: str | None,
) -> None:
    slots = _request_slots_for_call_type(data, call_type, mask_system)
    for getter, setter in slots:
        text = getter()
        if text:
            masked = await engine_mask(
                text,
                detector=detector,
                vault=vault,
                org_key=org_key,
                scope=scope,
                placeholder_format=placeholder_format,
                request_id=request_id,
            )
            setter(masked)


async def _remask_response(
    response: Any,
    detector: Detector,
    vault: MappingVault,
    org_key: str,
    scope: str,
    placeholder_format: str,
    request_id: str | None,
) -> None:
    # Try each adapter; each returns slots only for shapes it recognizes.
    for slots_fn in (
        openai_adapter.response_slots,
        anthropic_adapter.response_slots,
        responses_adapter.response_slots,
    ):
        try:
            slots = slots_fn(response)
        except Exception:  # noqa: BLE001 - never let logging re-mask crash the call
            continue
        for getter, setter in slots:
            text = getter()
            if text:
                masked = await engine_mask(
                    text,
                    detector=detector,
                    vault=vault,
                    org_key=org_key,
                    scope=scope,
                    placeholder_format=placeholder_format,
                    request_id=request_id,
                )
                setter(masked)


def _request_slots_for_call_type(data: dict, call_type: str, mask_system: bool) -> list:
    if call_type in {"responses", "aresponses"}:
        return responses_adapter.request_input_slots(data)
    if call_type in {"anthropic_messages"}:
        return anthropic_adapter.request_message_slots(data, mask_system=mask_system)
    # Default: OpenAI chat-completions shape.
    return openai_adapter.request_message_slots(data, mask_system=mask_system)
