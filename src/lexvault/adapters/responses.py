"""Responses API adapter — extract/replace text from ``/v1/responses`` shapes.

Best-effort (IMPLICIT_SPEC invariant 21): mask the request ``input`` text and
restore the response ``output_text``. The Responses API ``input`` may be a plain
string or a list of input items (each with ``content`` text parts); the response
``output`` is a list of items with ``content`` parts. On unknown shapes the
adapter returns no slots and the guardrail no-ops gracefully (no leak).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["request_input_slots", "response_slots"]

TextSlot = tuple[Callable[[], str], Callable[[str], None]]


def request_input_slots(data: dict) -> list[TextSlot]:
    """Mutable text slots for a ``/v1/responses`` request ``input`` field."""
    inp = data.get("input")
    if isinstance(inp, str):
        return [_make_input_str_slot(data)]
    if isinstance(inp, list):
        slots: list[TextSlot] = []
        for item in inp:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        slots.append(_slot_str(part, "text"))
            elif isinstance(item.get("content"), str):
                slots.append(_slot_str(item, "content"))
        return slots
    return []


def response_slots(response: Any) -> list[TextSlot]:
    """Mutable text slots for a ``/v1/responses`` response (best-effort).

    Handles ``output_text`` (str), and ``output`` (list of items with content
    parts). No-ops gracefully on unknown shapes.
    """
    slots: list[TextSlot] = []

    # output_text convenience field (str).
    ot = _get_attr(response, "output_text")
    if isinstance(ot, str) and ot:
        slots.append(_attr_slot(response, "output_text"))

    # output is a list of message items with content parts.
    output = _get_attr(response, "output")
    if isinstance(output, list):
        for item in output:
            content = _get_attr(item, "content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        slots.append(_slot_str(part, "text"))
            elif isinstance(content, str):
                slots.append(_attr_slot(item, "content"))
    return slots


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


def _make_input_str_slot(data: dict) -> TextSlot:
    def getter() -> str:
        return str(data.get("input", "") or "")

    def setter(value: str) -> None:
        data["input"] = value

    return getter, setter


def _get_attr(obj: Any, attr: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)
