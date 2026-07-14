"""Streaming restore dispatcher.

Ties the SSE re-framer (:mod:`reframer`) and the restore-aware bounded buffer
(:mod:`buffer`) to the two streaming regimes from IMPLICIT_SPEC invariant 6:

  - **Parsed-object regime** — OpenAI ``ModelResponseStream`` chunks and
    ``ResponsesAPIStreamingResponse`` chunks. Each chunk is a Python object with
    a ``.delta.content`` string; we accumulate text, restore placeholders once
    they've left the placeholder boundary, and re-emit chunks.
  - **Raw-bytes regime** — Anthropic-native ``/v1/messages`` yields raw SSE
    ``bytes`` split at arbitrary boundaries; we re-frame via :class:`SseReframer`,
    extract text from the Anthropic delta types, accumulate, restore, and
    re-emit as restored ``text_delta`` frames.

The correctness core (RC4 fix): a placeholder can span multiple chunks OR
multiple SSE frames. We therefore accumulate ALL text into a restore-aware
buffer that only emits text once it has provably left any placeholder boundary,
so ``unmask`` always sees complete placeholders. Fail-closed (invariant 11): on
a dangling unclosed opener at stream end, we signal an error frame.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from lexvault.streaming.buffer import PlaceholderBuffer
from lexvault.streaming.reframer import SseReframer, extract_frame_text, restore_frame_text
from lexvault.vault import MappingVault

__all__ = ["streaming_restore", "StreamingError"]


class StreamingError(RuntimeError):
    """Raised to signal the caller should emit an error frame and stop the stream."""


async def streaming_restore(
    upstream: AsyncIterator[Any],
    *,
    vault: MappingVault,
    placeholder_namespace_re: str,
    max_placeholder_len: int,
) -> AsyncIterator[Any]:
    """Wrap an upstream chunk iterator, restoring placeholders in each chunk.

    Dispatches on the type of the first chunk (invariant 6): ``bytes`` →
    raw-SSE reframer path; any object → parsed path. Yields restored chunks of
    the same kind.

    On a dangling partial placeholder at stream end: fail-closed — raise
    :class:`StreamingError` so the guardrail can emit an error frame.
    """
    peeked: list[Any] = []
    first: Any = None

    it = upstream.__aiter__()
    exhausted = False
    try:
        first = await it.__anext__()
    except StopAsyncIteration:
        exhausted = True
    if exhausted:
        return
        yield  # pragma: no cover - make this an async generator for typing

    if isinstance(first, (bytes, bytearray)):
        peeked.append(first)
        async for out in _restore_bytes(
            _chain(peeked, it),
            vault=vault,
            namespace_re=placeholder_namespace_re,
            window=max_placeholder_len,
        ):
            yield out
    else:
        peeked.append(first)
        async for out in _restore_objects(
            _chain(peeked, it),
            vault=vault,
            namespace_re=placeholder_namespace_re,
            window=max_placeholder_len,
        ):
            yield out


async def _chain(seed: list[Any], it: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Yield seed items then drain ``it``."""
    for item in seed:
        yield item
    async for item in it:
        yield item


# --------------------------------------------------------------------------- #
# Raw-SSE-bytes regime (Anthropic /v1/messages)
# --------------------------------------------------------------------------- #
async def _restore_bytes(
    upstream: AsyncIterator[Any],
    *,
    vault: MappingVault,
    namespace_re: str,
    window: int,
) -> AsyncIterator[bytes]:
    """Restore placeholders in an Anthropic raw-SSE-bytes stream, IN PLACE.

    CS-E3: each frame is restored preserving its delta type (text_delta stays
    text_delta, thinking_delta stays thinking_delta, input_json_delta stays
    input_json_delta) and its content-block index, and structural frames
    (content_block_start/stop, message_start/stop, ping) pass through in order.
    A placeholder can straddle frames within the SAME (index, type), so we keep
    a per-(index, json-pointer) PlaceholderBuffer — text never crosses into
    thinking or tool-arg frames, so type fidelity is preserved.
    """
    reframer = SseReframer()
    bufs: dict[tuple[int, str], PlaceholderBuffer] = {}
    # Track whether ANY buffer ended mid-placeholder → fail-closed signal.
    had_partial = False

    def _buf_for(index: int, pointer: str) -> PlaceholderBuffer:
        key = (index, pointer)
        b = bufs.get(key)
        if b is None:
            b = PlaceholderBuffer(window, namespace_re)
            bufs[key] = b
        return b

    async for raw in upstream:
        reframer.feed(bytes(raw))
        for frame in reframer.drain_complete():
            slots = extract_frame_text(frame)
            if not slots:
                # Non-restorable frame (signature_delta, message_delta, ping) or
                # a content_block_start with empty/non-text content. Pass through
                # verbatim so structural ordering + block lifecycle is preserved.
                # CS-E3: this keeps empty-text content_block_start alive.
                yield frame.raw
                continue
            # Restore each slot's text through its own (index, pointer) buffer so
            # a placeholder split across frames of the same type/index reassembles,
            # and write the restored text back into THIS frame in place.
            restorations: dict[str, str] = {}
            for pointer, text in slots:
                index = _frame_index(frame)
                buf = _buf_for(index, pointer)
                buf.feed(text)
                try:
                    ready = buf.drain_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
                except Exception as exc:  # noqa: BLE001 - CS-E5: route to error frame
                    msg = f"restore failed mid-stream: {exc}"
                    raise StreamingError(msg) from exc
                restorations[pointer] = ready  # may be "" if the window holds it
            yield restore_frame_text(frame, restorations)

    # Flush every per-(index,type) buffer on stream end.
    for (index, pointer), buf in bufs.items():
        try:
            tail, partial_in_ns = buf.flush_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001 - CS-E5
            msg = f"restore failed during stream flush: {exc}"
            raise StreamingError(msg) from exc
        if partial_in_ns:
            had_partial = True
        if tail and not partial_in_ns:
            # Emit a final delta of the matching type for the held tail.
            yield _tail_delta_bytes(index, pointer, tail)
    if had_partial:
        msg = "stream ended with a dangling partial placeholder"
        raise StreamingError(msg)


def _frame_index(frame: Any) -> int:
    """The content-block index for an Anthropic frame, defaulting to 0."""
    if frame.data_obj and isinstance(frame.data_obj.get("index"), int):
        return frame.data_obj["index"]
    return 0


def _tail_delta_bytes(index: int, pointer: str, text: str) -> bytes:
    """Re-emit a held-tail restore as a delta frame matching ``pointer``'s type.

    CS-E3: preserves the delta type so a thinking tail stays thinking, a
    tool-arg tail stays input_json_delta, and a text tail stays text_delta.
    """
    if pointer == "delta.thinking":
        delta: dict = {"type": "thinking_delta", "thinking": text}
    elif pointer == "delta.partial_json":
        delta = {"type": "input_json_delta", "partial_json": text}
    elif pointer in ("content_block.text", "content_block.thinking"):
        # content_block_start tail: the block already opened; emit a delta of
        # the matching type (text_delta / thinking_delta).
        is_thinking = pointer.endswith("thinking")
        kind = "thinking_delta" if is_thinking else "text_delta"
        field = "thinking" if is_thinking else "text"
        delta = {"type": kind, field: text}
    else:
        delta = {"type": "text_delta", "text": text}
    payload = {"type": "content_block_delta", "index": index, "delta": delta}
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


def _text_delta_bytes(text: str, index: int) -> bytes:
    """Re-emit restored text as an Anthropic content_block_delta text_delta frame."""
    payload = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()


# --------------------------------------------------------------------------- #
# Parsed-object regime (OpenAI ModelResponseStream / ResponsesAPIStreamingResponse)
# --------------------------------------------------------------------------- #
async def _restore_objects(
    upstream: AsyncIterator[Any],
    *,
    vault: MappingVault,
    namespace_re: str,
    window: int,
) -> AsyncIterator[Any]:
    """Restore text in parsed streaming objects (OpenAI ModelResponseStream deltas).

    A placeholder can span multiple delta chunks, so we accumulate the text
    stream, hold back the trailing window (≤ longest placeholder), and restore
    whole placeholders. This satisfies invariant 5 (bounded buffering) and
    invariant 1 (a split placeholder still restores).

    CS-E6: we MUTATE the original chunk objects in place — writing restored text
    into each chunk's delta content — instead of synthesizing SimpleNamespace
    chunks. This preserves the ``ModelResponseStream`` metadata (id/model/
    created/object/finish_reason) that LiteLLM's proxy serializer relies on, and
    handles all choices. Restored text is carried in a pending buffer between
    chunks (the window may delay a placeholder's restore past its source chunk).
    """
    buf = PlaceholderBuffer(window, namespace_re)
    pending = ""  # restored text not yet written into a yielded chunk

    async for chunk in upstream:
        slots = _extract_object_text(chunk)
        # Feed this chunk's text into the restore buffer; accumulate what's
        # ready (restored) into the pending string.
        chunk_text = "".join(getter() for getter, _ in slots)
        if chunk_text:
            buf.feed(chunk_text)
        ready = buf.drain_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
        if ready:
            pending += ready

        if not slots:
            # Non-text chunk (usage/finish): pass through verbatim. Its text
            # (already restored into `pending`) is written to the NEXT content
            # chunk, preserving ordering. Yield the original non-text chunk.
            yield chunk
            continue

        # Write the pending restored text into this chunk's slots (mutating in
        # place across ALL choices), then yield the original chunk. Empty
        # pending means the window is still holding this chunk's text; we still
        # yield the chunk with its content cleared so the stream isn't reordered
        # and metadata reaches the client. The held text restores on a later chunk.
        _write_object_slots(slots, pending)
        pending = ""
        yield chunk

    # Flush the held window on stream end.
    # CS-E5: route a VaultError through the error-frame path.
    try:
        tail, partial_in_ns = buf.flush_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001 - route any restore error to error-frame path
        msg = f"restore failed during stream flush: {exc}"
        raise StreamingError(msg) from exc
    final_text = pending + (tail if not partial_in_ns else "")
    # CS-E2: never yield a dangling partial placeholder (partial_in_ns suppresses tail).
    if final_text:
        # No further content chunk to mutate — synthesize one carrying the
        # restored tail. This is the only synthesized chunk and only on stream
        # end, so metadata loss here is unavoidable and acceptable (the upstream
        # stream is done; no original chunk remains to carry the tail).
        yield _make_text_chunk(final_text)
    if partial_in_ns:
        msg = "stream ended with a dangling partial placeholder"
        raise StreamingError(msg)


def _extract_object_text(chunk: Any) -> list[tuple[Any, Any]]:
    """Return [(getter, setter)] callables for each restorable text field in ``chunk``.

    Handles OpenAI ``ModelResponseStream`` (``choices[].delta.content`` as str)
    and best-effort ``ResponsesAPIStreamingResponse`` shapes. CS-E6: ALL choices
    are walked (not just ``choices[0]``).
    """
    slots: list[tuple[Any, Any]] = []

    choices = getattr(chunk, "choices", None)
    if isinstance(choices, list) and choices:
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                # Include empty-string content too so we can write restored text
                # into it; only skip when there is no delta.content attribute.
                if isinstance(content, str):
                    slots.append(_make_attr_slot(delta, "content"))

    if isinstance(chunk, dict):
        delta = chunk.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            slots.append(_make_dict_slot(delta, "content"))
        if isinstance(chunk.get("text"), str):
            slots.append(_make_dict_slot(chunk, "text"))

    return slots


def _write_object_slots(slots: list[tuple[Any, Any]], text: str) -> None:
    """Write restored ``text`` into ``slots``, mutating objects in place (CS-E6).

    All restored text goes into the FIRST slot (the client concatenates delta
    contents in order). Remaining content slots are cleared (their text was
    already fed to the restore buffer and will re-emerge on a later chunk).
    """
    if not slots:
        return
    first_getter, first_setter = slots[0]
    first_setter(text)
    for _getter, setter in slots[1:]:
        setter("")


def _make_attr_slot(obj: Any, attr: str) -> tuple[Any, Any]:
    def getter() -> str:
        return str(getattr(obj, attr, "") or "")

    def setter(value: str) -> None:
        setattr(obj, attr, value)

    return getter, setter


def _make_dict_slot(d: dict, key: str) -> tuple[Any, Any]:
    def getter() -> str:
        return str(d.get(key, "") or "")

    def setter(value: str) -> None:
        d[key] = value

    return getter, setter


def _make_text_chunk(content: str) -> Any:
    """Build a minimal OpenAI ModelResponseStream-shaped chunk with text content."""
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=None)]
    )


# --------------------------------------------------------------------------- #
# Error-frame helpers (used by the guardrail on StreamingError)
# --------------------------------------------------------------------------- #
def openai_error_chunk(message: str) -> bytes:
    """A minimal OpenAI-style streaming error data line."""
    payload = json.dumps({"error": {"message": message, "type": "server_error"}})
    return f"data: {payload}\n\n".encode()


def anthropic_error_event(message: str) -> bytes:
    """A minimal Anthropic SSE ``event: error`` frame."""
    payload = json.dumps({"type": "error", "error": {"type": "api_error", "message": message}})
    return f"event: error\ndata: {payload}\n\n".encode()
