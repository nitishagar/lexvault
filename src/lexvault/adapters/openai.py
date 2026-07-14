"""OpenAI adapter — extract/replace text from OpenAI request/response shapes.

Handles:
  - Request ``messages``: ``system``/``user``/``assistant`` roles with ``content``
    (str or list of ``{"type":"text","text":...}`` parts), and assistant messages
    carrying ``tool_calls[].function.arguments`` (a JSON string).
  - Response ``ModelResponse``: ``choices[].message.content`` (str or parts list)
    and ``choices[].message.tool_calls[].function.arguments`` (JSON string).

Text slots are returned as ``(getter, setter)`` closures so the caller can
mask/unmask the text in place without the adapter needing to know the engine.

Tool *definitions* in ``data["tools"]`` are deliberately NOT touched
(IMPLICIT_SPEC invariant 21 — the model needs verbatim schemas for dispatch).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

__all__ = ["request_message_slots", "response_slots", "tool_call_argument_slots"]

# A TextSlot is a (getter, setter) pair: getter() -> str, setter(str) -> None.
TextSlot = tuple[Callable[[], str], Callable[[str], None]]


def request_message_slots(data: dict, *, mask_system: bool = True) -> list[TextSlot]:
    """Return mutable text slots for an OpenAI chat-completions request body.

    Covers ``data["messages"]`` content (gated by ``mask_system`` for the system
    role) and in-history ``tool_calls[].function.arguments``.
    """
    slots: list[TextSlot] = []
    messages = data.get("messages")
    if not isinstance(messages, list):
        return slots

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system" and not mask_system:
            continue
        # content: str OR list of {"type":"text","text":...}
        content = msg.get("content")
        if isinstance(content, str):
            slots.append(_slot_str(msg, "content"))
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    slots.append(_slot_str(part, "text"))
        # In-history assistant tool_calls arguments.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            slots.extend(_tool_call_args_slots(tool_calls))
    return slots


def response_slots(response: Any) -> list[TextSlot]:
    """Return mutable text slots for an OpenAI ``ModelResponse``.

    ``choices[].message.content`` (str or parts) and
    ``choices[].message.tool_calls[].function.arguments``.
    """
    slots: list[TextSlot] = []
    choices = _get_choices(response)
    for choice in choices:
        message = _get_attr(choice, "message")
        if message is None:
            continue
        content = _get_attr(message, "content")
        if isinstance(content, str):
            slots.append(_attr_slot(message, "content"))
        elif isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ):
                    slots.append(_slot_str(part, "text"))
        tool_calls = _get_attr(message, "tool_calls")
        if isinstance(tool_calls, list):
            slots.extend(_tool_call_obj_slots(tool_calls))
    return slots


def tool_call_argument_slots(data: dict) -> list[TextSlot]:
    """Slots for top-level ``data["tool_calls"]`` if present (request-side helper)."""
    tc = data.get("tool_calls")
    if isinstance(tc, list):
        return _tool_call_args_slots(tc)
    return []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _slot_str(obj: dict, key: str) -> TextSlot:
    def getter() -> str:
        return str(obj.get(key, "") or "")

    def setter(value: str) -> None:
        obj[key] = value

    return getter, setter


def _attr_slot(obj: Any, attr: str) -> TextSlot:
    def getter() -> str:
        return str(getattr(obj, attr, "") or "")

    def setter(value: str) -> None:
        setattr(obj, attr, value)

    return getter, setter


def _tool_call_args_slots(tool_calls: list) -> list[TextSlot]:
    """Slots for tool_calls entries that are plain dicts with ``function.arguments`` str."""
    slots: list[TextSlot] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
            slots.append(_slot_str(fn, "arguments"))
    return slots


def _tool_call_obj_slots(tool_calls: list) -> list[TextSlot]:
    """Slots for tool_calls that may be objects (litellm Message.tool_calls)."""
    slots: list[TextSlot] = []
    for tc in tool_calls:
        fn = _get_attr(tc, "function")
        if fn is None:
            continue
        args = _get_attr(fn, "arguments")
        if isinstance(args, str):
            slots.append(_attr_slot(fn, "arguments"))
        elif isinstance(args, dict):
            # The model may return structured args; mask each string value.
            slots.extend(_dict_str_value_slots(args))
    return slots


def _dict_str_value_slots(d: dict) -> list[TextSlot]:
    """A slot for each string value in a (flat) dict — used for structured tool args."""
    slots: list[TextSlot] = []
    for k, v in d.items():
        if isinstance(v, str):
            slots.append(_make_dict_slot(d, k))
    return slots


def _make_dict_slot(d: dict, key: str) -> TextSlot:
    def getter() -> str:
        return str(d.get(key, "") or "")

    def setter(value: str) -> None:
        d[key] = value

    return getter, setter


def _get_choices(response: Any) -> list:
    choices = _get_attr(response, "choices")
    return choices if isinstance(choices, list) else []


def _get_attr(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def parse_arguments(arguments: str) -> object:
    """Parse a JSON ``arguments`` string; tolerate empty string as {}."""
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        # Some providers emit partial JSON during streaming; pass through.
        return arguments
