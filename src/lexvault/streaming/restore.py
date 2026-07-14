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
from lexvault.streaming.reframer import SseReframer, extract_frame_text
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
    reframer = SseReframer()
    buf = PlaceholderBuffer(window, namespace_re)
    last_index = 0

    async for raw in upstream:
        reframer.feed(bytes(raw))
        for frame in reframer.drain_complete():
            slots = extract_frame_text(frame)
            if not slots:
                # Non-restorable frame (signature_delta, message_delta, ping) —
                # flush any restored text first, then pass through verbatim.
                ready = buf.drain_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
                if ready:
                    yield _text_delta_bytes(ready, last_index)
                yield frame.raw
                # Track the content-block index for re-emitted text deltas.
                if frame.data_obj and isinstance(frame.data_obj.get("index"), int):
                    last_index = frame.data_obj["index"]
                continue
            # Accumulate ALL extracted text; the buffer restores complete
            # placeholders and only emits text past the placeholder boundary.
            for _pointer, text in slots:
                buf.feed(text)
            ready = buf.drain_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
            if ready:
                yield _text_delta_bytes(ready, last_index)

    # Flush on stream end — restore complete placeholders in the held tail.
    tail, partial_in_ns = buf.flush_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
    if tail:
        yield _text_delta_bytes(tail, last_index)
    if partial_in_ns:
        msg = "stream ended with a dangling partial placeholder"
        raise StreamingError(msg)


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
    stream, hold back the trailing window (≤ longest placeholder), restore whole
    placeholders, and re-emit as chunks. This satisfies invariant 5 (bounded
    buffering) and invariant 1 (a split placeholder still restores).
    """
    buf = PlaceholderBuffer(window, namespace_re)

    async for chunk in upstream:
        slots = _extract_object_text(chunk)
        if not slots:
            # Non-text chunk (e.g. usage/finish) — flush restored text first so we
            # don't reorder, then pass the chunk through verbatim.
            ready = buf.drain_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
            if ready:
                yield _make_text_chunk(ready)
            yield chunk
            continue

        chunk_text = "".join(getter() for getter, _ in slots)
        buf.feed(chunk_text)
        ready = buf.drain_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
        if ready:
            yield _make_text_chunk(ready)

    # Flush the held window on stream end.
    tail, partial_in_ns = buf.flush_restored_with_vault(vault._lookup_sync)  # noqa: SLF001
    if tail:
        yield _make_text_chunk(tail)
    if partial_in_ns:
        msg = "stream ended with a dangling partial placeholder"
        raise StreamingError(msg)


def _extract_object_text(chunk: Any) -> list[tuple[Any, Any]]:
    """Return [(getter, setter)] callables for each restorable text field in ``chunk``.

    Handles OpenAI ``ModelResponseStream`` (``choices[0].delta.content`` as str)
    and best-effort ``ResponsesAPIStreamingResponse`` shapes.
    """
    slots: list[tuple[Any, Any]] = []

    choices = getattr(chunk, "choices", None)
    if isinstance(choices, list) and choices:
        delta = getattr(choices[0], "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                slots.append(_make_attr_slot(delta, "content"))

    if isinstance(chunk, dict):
        delta = chunk.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            slots.append(_make_dict_slot(delta, "content"))
        if isinstance(chunk.get("text"), str):
            slots.append(_make_dict_slot(chunk, "text"))

    return slots


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
