"""Anthropic-native adapter — extract/replace text from Anthropic shapes.

Handles:
  - Request ``messages`` content blocks: ``{"type":"text","text":...}`` and
    assistant ``{"type":"tool_use","input": {...}}`` (mask each string value in
    ``input``). The top-level ``system`` prompt (str or list of text blocks) is
    masked, gated by ``mask_system``.
  - Response (``AnthropicMessagesResponse`` or content-block dict): ``content``
    blocks of ``type:"text"`` (``.text``) and ``type:"tool_use"`` (``.input`` dict
    → mask each string value).

Tool *definitions* in ``data["tools"]`` are NOT touched (invariant 21).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["request_message_slots", "response_slots"]

TextSlot = tuple[Callable[[], str], Callable[[str], None]]


def request_message_slots(data: dict, *, mask_system: bool = True) -> list[TextSlot]:
    """Mutable text slots for an Anthropic-native request body (``/v1/messages``)."""
    slots: list[TextSlot] = []
    if mask_system:
        system = data.get("system")
        if isinstance(system, str) and system:
            # String system prompts are rewritten via data["system"]; the slot
            # setter writes the whole string back to the request body.
            slots.append(_make_system_str_slot(data))
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    slots.append(_slot_str(block, "text"))

    messages = data.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            slots.extend(_content_blocks_slots(content))
    return slots


def response_slots(response: Any) -> list[TextSlot]:
    """Mutable text slots for an Anthropic-native response.

    Works with both an ``AnthropicMessagesResponse`` object (``.content`` list of
    blocks with ``.text``/``.input``) and a plain dict (``{"content":[...]}``).
    """
    content = _get_attr(response, "content")
    return _content_blocks_slots(content)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _content_blocks_slots(content: Any) -> list[TextSlot]:
    if not isinstance(content, list):
        return []
    slots: list[TextSlot] = []
    for block in content:
        if not isinstance(block, dict):
            # litellm objects: try attribute access.
            btype = _get_attr(block, "type")
            if btype == "text":
                text = _get_attr(block, "text")
                if isinstance(text, str):
                    slots.append(_attr_slot(block, "text"))
            elif btype == "tool_use":
                inp = _get_attr(block, "input")
                if isinstance(inp, dict):
                    slots.extend(_dict_str_value_slots(inp))
            continue
        btype = block.get("type")
        if btype == "text" and isinstance(block.get("text"), str):
            slots.append(_slot_str(block, "text"))
        elif btype == "tool_use" and isinstance(block.get("input"), dict):
            slots.extend(_dict_str_value_slots(block["input"]))
    return slots


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


def _make_system_str_slot(data: dict) -> TextSlot:
    def getter() -> str:
        return str(data.get("system", "") or "")

    def setter(value: str) -> None:
        data["system"] = value

    return getter, setter


def _dict_str_value_slots(d: dict) -> list[TextSlot]:
    slots: list[TextSlot] = []
    for k, v in list(d.items()):
        if isinstance(v, str):
            slots.append(_make_dict_slot(d, k))
    return slots


def _make_dict_slot(d: dict, key: str) -> TextSlot:
    def getter() -> str:
        return str(d.get(key, "") or "")

    def setter(value: str) -> None:
        d[key] = value

    return getter, setter


def _get_attr(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)
