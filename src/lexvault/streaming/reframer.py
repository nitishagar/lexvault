"""SSE byte re-framer + Anthropic delta text extractor.

On ``/v1/messages`` with the native Anthropic provider, LiteLLM hands the
streaming iterator hook **raw SSE ``bytes``** split at arbitrary (non-frame)
boundaries by httpx ``aiter_bytes()`` (IMPLICIT_SPEC invariant 6). This module:

  1. Accumulates raw bytes and emits complete SSE *frames* (an ``event:`` line
     plus its ``data:`` JSON payload, terminated by a blank line).
  2. For each frame, extracts the restorable text from every Anthropic delta
     type that can carry a masked codename:
       - ``content_block_delta`` / ``text_delta``  → ``.delta.text``
       - ``content_block_delta`` / ``thinking_delta`` → ``.delta.thinking``
         (extended thinking CAN echo masked codenames — restored too)
       - ``content_block_delta`` / ``input_json_delta`` → ``.delta.partial_json``,
         accumulated per content-block ``index`` into a JSON string; restored as
         a whole string before re-emitting (tool-call argument increments).
       - ``content_block_start`` → initial ``text``/``thinking`` on the block.
       - ``signature_delta`` and ``message_delta`` (stop_reason/usage) carry NO
         restorable text and pass through untouched.

The restore layer calls :meth:`SseReframer.feed` with raw bytes and receives
``(restored_bytes, raw_emitted_bytes)`` — restored bytes are frames whose text
was masked back; raw_emitted_bytes are frames that needed no transformation,
yielded verbatim so the client sees byte-identical SSE otherwise.
"""

from __future__ import annotations

import json
import re

__all__ = ["SseReframer", "extract_frame_text", "restore_frame_text", "Frame"]


class Frame:
    """A single complete SSE frame: the raw bytes plus parsed JSON data (if any).

    ``data_obj`` is the parsed ``data:`` JSON dict, or None if the frame had no
    parseable JSON (e.g. an ``event: ping`` with no data). ``raw`` is the exact
    original bytes of the frame (including the trailing ``\\n\\n``) so frames
    that need no restore can be re-emitted byte-identically.
    """

    __slots__ = ("raw", "event", "data_obj", "data_raw")

    def __init__(
        self, raw: bytes, event: str | None, data_raw: str | None, data_obj: dict | None
    ) -> None:
        self.raw = raw
        self.event = event
        self.data_raw = data_raw
        self.data_obj = data_obj


# A frame is terminated by a blank line. We split on the first ``\n\n``.
_FRAME_SPLIT = re.compile(rb"\r?\n\r?\n")


class SseReframer:
    """Accumulate raw SSE bytes; yield complete frames one at a time.

    Usage::

        reframer = SseReframer()
        reframer.feed(b"...raw bytes...")
        for frame in reframer.drain_complete():
            ...
        # On stream end:
        leftover = reframer.remaining()
    """

    def __init__(self) -> None:
        self._buf = b""

    def feed(self, data: bytes) -> None:
        if data:
            self._buf += data

    def drain_complete(self) -> list[Frame]:
        """Pop and return all fully-terminated frames currently buffered.

        Each returned Frame's ``raw`` includes its terminating blank line. The
        buffer keeps any trailing partial frame.
        """
        frames: list[Frame] = []
        while True:
            m = _FRAME_SPLIT.search(self._buf)
            if m is None:
                break
            raw = self._buf[: m.end()]
            self._buf = self._buf[m.end() :]
            frame = _parse_frame(raw)
            if frame is not None:
                frames.append(frame)
        return frames

    def remaining(self) -> bytes:
        """Return any un-terminated trailing bytes (call on stream end)."""
        rest = self._buf
        self._buf = b""
        return rest


def _parse_frame(raw: bytes) -> Frame | None:
    """Parse a raw SSE frame into event + data. Returns None for empty/comment frames."""
    text = raw.decode("utf-8", errors="replace")
    event: str | None = None
    data_lines: list[str] = []
    for line in text.splitlines():
        if not line or line.startswith(":"):  # blank line (terminator) or comment
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    data_raw = "\n".join(data_lines) if data_lines else None
    data_obj: dict | None = None
    if data_raw:
        try:
            parsed = json.loads(data_raw)
            data_obj = parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            data_obj = None
    return Frame(raw=raw, event=event, data_raw=data_raw, data_obj=data_obj)


# --------------------------------------------------------------------------- #
# Delta text extraction / restoration
# --------------------------------------------------------------------------- #


def extract_frame_text(frame: Frame) -> list[tuple[str, str]]:
    """Return a list of (json_pointer, text) restorable strings in this frame.

    ``json_pointer`` is a dotted path into ``frame.data_obj`` identifying the
    field the text came from, so :func:`restore_frame_text` can write the
    restored value back. Paths used:
      - ``delta.text``        (content_block_delta / text_delta)
      - ``delta.thinking``    (content_block_delta / thinking_delta)
      - ``delta.partial_json`` (content_block_delta / input_json_delta — tool args)
      - ``content.text``      (content_block_start initial text)
      - ``content.thinking``  (content_block_start initial thinking)
    """
    out: list[tuple[str, str]] = []
    data = frame.data_obj
    if not isinstance(data, dict):
        return out

    frame_type = data.get("type")

    if frame_type == "content_block_delta":
        delta = data.get("delta")
        if isinstance(delta, dict):
            dt = delta.get("type")
            if dt == "text_delta" and isinstance(delta.get("text"), str):
                out.append(("delta.text", delta["text"]))
            elif dt == "thinking_delta" and isinstance(delta.get("thinking"), str):
                out.append(("delta.thinking", delta["thinking"]))
            elif dt == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                out.append(("delta.partial_json", delta["partial_json"]))
        return out

    if frame_type == "content_block_start":
        block = data.get("content_block")
        if isinstance(block, dict):
            if isinstance(block.get("text"), str):
                out.append(("content_block.text", block["text"]))
            if isinstance(block.get("thinking"), str):
                out.append(("content_block.thinking", block["thinking"]))
        return out

    return out


def restore_frame_text(frame: Frame, restorations: dict[str, str]) -> bytes:
    """Return re-serialized SSE bytes for ``frame`` with the given text restored.

    ``restorations`` maps each json-pointer from :func:`extract_frame_text` to
    its restored text. The frame's JSON ``data:`` line is rewritten in place;
    non-data lines (``event:``, comments) and the blank-line terminator are
    preserved. Frames with no restorable text should be re-emitted via
    ``frame.raw`` instead (caller decides).
    """
    if not restorations or frame.data_obj is None:
        return frame.raw

    data = frame.data_obj
    for pointer, new_text in restorations.items():
        _set_by_pointer(data, pointer, new_text)

    # Re-serialize: replace the data line(s) with the new JSON, keep other lines.
    new_data_json = json.dumps(data, separators=(",", ":"))
    text = frame.raw.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    data_written = False
    for line in lines:
        if line.startswith("data:") and not data_written:
            # Replace the first data: line; drop subsequent contiguous data: lines.
            out.append(f"data: {new_data_json}\n")
            data_written = True
        elif line.startswith("data:") and data_written:
            # Swallow extra data: lines that were part of the original multi-line payload.
            continue
        else:
            out.append(line)
    if not data_written:
        # The frame had a data_obj but no explicit data: line (shouldn't happen);
        # append one before the terminating blank line.
        if out and out[-1].strip() == "":
            out.insert(len(out) - 1, f"data: {new_data_json}\n")
        else:
            out.append(f"data: {new_data_json}\n")
    return "".join(out).encode("utf-8")


def _set_by_pointer(data: dict, pointer: str, value: str) -> None:
    """Set a nested field in ``data`` by a dotted pointer (top 2 levels only)."""
    parts = pointer.split(".")
    if len(parts) == 2:
        container = data.get(parts[0])
        if isinstance(container, dict):
            container[parts[1]] = value
    elif len(parts) == 1:
        data[parts[0]] = value
