"""Unit tests for the streaming restore layer (buffer + reframer + dispatcher).

These are pure-logic tests (no LiteLLM) pinning IMPLICIT_SPEC invariants:
  5 (no whole-response buffering), 6 (two streaming regimes — bytes reframer +
  parsed objects), 1 (round-trip across chunk splits), 11 (fail-closed on a
  dangling partial placeholder).

The plan's named edge cases: placeholder split at chunk boundary; placeholder
spanning 3+ chunks; empty stream; stream ending mid-placeholder; mid-frame byte
splits; Anthropic thinking_delta restored (not just text_delta).
"""

from __future__ import annotations

import json

import pytest

from lexvault.streaming.buffer import PlaceholderBuffer
from lexvault.streaming.reframer import SseReframer, extract_frame_text, restore_frame_text
from lexvault.streaming.restore import (
    StreamingError,
    anthropic_error_event,
    openai_error_chunk,
    streaming_restore,
)
from lexvault.vault import MappingVault

NS = r"\[LEX\-[A-Z2-7]+\](?:-\d+)?"
WINDOW = 20  # generous for [LEX-XXXXXXXX]-9999


# --------------------------------------------------------------------------- #
# PlaceholderBuffer — bounded window, split-across-boundaries (inv 5)
# --------------------------------------------------------------------------- #
def _noop_lookup(_ph: str) -> str | None:
    return None


class TestBuffer:
    def test_emits_immediately_when_beyond_window(self):
        buf = PlaceholderBuffer(WINDOW, NS)
        buf.feed("x" * 30)
        ready = buf.drain_restored_with_vault(_noop_lookup)
        assert len(ready) == 30 - WINDOW  # only the trailing window is held

    def test_holds_everything_within_window(self):
        buf = PlaceholderBuffer(WINDOW, NS)
        buf.feed("short")
        assert buf.drain_restored_with_vault(_noop_lookup) == ""  # all held (≤ window)

    def test_placeholder_split_across_two_feeds_reassembled(self):
        """A placeholder split at a feed boundary is not emitted in halves."""
        placeholder = "[LEX-AAAAAAAA]"

        # A lookup that resolves the placeholder to "Original".
        def lookup(ph: str) -> str | None:
            return "Original" if ph == placeholder else None

        buf = PlaceholderBuffer(WINDOW, NS)
        first, second = placeholder[:7], placeholder[7:] + " trailing text here pad"
        buf.feed("text " + first)
        assert buf.drain_restored_with_vault(lookup) == ""  # nothing safe to emit yet
        buf.feed(second)
        ready = buf.drain_restored_with_vault(lookup)
        remaining, partial = buf.flush_restored_with_vault(lookup)
        full = ready + remaining
        # The placeholder was restored whole (never split, never leaked).
        assert "Original" in full
        assert placeholder not in full
        assert not partial

    def test_placeholder_spanning_three_feeds(self):
        placeholder = "[LEX-BBBBBBBB]"

        def lookup(ph: str) -> str | None:
            return "Restored" if ph == placeholder else None

        buf = PlaceholderBuffer(WINDOW, NS)
        parts = ["pre ", placeholder[:4], placeholder[4:9], placeholder[9:] + " post padding pad"]
        out = ""
        for p in parts:
            buf.feed(p)
            out += buf.drain_restored_with_vault(lookup)
        out += buf.flush_restored_with_vault(lookup)[0]
        assert "Restored" in out
        assert placeholder not in out

    def test_split_at_window_boundary_restores(self):
        """The RC4 core case: a placeholder straddling the window cut restores.

        This is the exact failure the impl-validator caught: if unmask ran on
        each ready slice independently, a placeholder spanning the cut leaked.
        The restore-aware buffer emits text only once it has left the boundary.
        """
        placeholder = "[LEX-CCCCCCCC]"

        def lookup(ph: str) -> str | None:
            return "Codename" if ph == placeholder else None

        buf = PlaceholderBuffer(len(placeholder), NS)
        # Construct text so the placeholder straddles the window cut.
        # total = 8 (prefix) + 14 (placeholder) + 8 (suffix) = 30; window=14.
        # After feed, cut = 30-14 = 16 → ready = prefix(8) + placeholder[:8],
        # held = placeholder[8:] + suffix. The ready slice contains a HALF
        # placeholder — it must NOT be emitted unrestored.
        prefix = "xxxxxxxx"  # 8 chars
        suffix = "yyyyyyyy"  # 8 chars
        buf.feed(prefix + placeholder + suffix)
        ready = buf.drain_restored_with_vault(lookup)
        remaining, _ = buf.flush_restored_with_vault(lookup)
        full = ready + remaining
        assert "Codename" in full
        assert placeholder not in full  # never leaked unrestored

    def test_flush_partial_in_namespace_when_dangling_placeholder(self):
        """Stream ending mid-placeholder → partial_in_namespace=True (fail-closed signal)."""
        buf = PlaceholderBuffer(WINDOW, NS)
        buf.feed("[LEX-AAA")  # a partial placeholder opener, no close
        remaining, partial = buf.flush_restored_with_vault(_noop_lookup)
        assert remaining == "[LEX-AAA"
        assert partial is True

    def test_flush_no_partial_for_plain_text(self):
        buf = PlaceholderBuffer(WINDOW, NS)
        buf.feed("plain trailing text")
        remaining, partial = buf.flush_restored_with_vault(_noop_lookup)
        assert partial is False
        assert remaining == "plain trailing text"


# --------------------------------------------------------------------------- #
# SseReframer — mid-frame byte splits (inv 6)
# --------------------------------------------------------------------------- #
class TestReframer:
    def _text_delta_frame(self, text: str, index: int = 0) -> bytes:
        payload = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
        return f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()

    def test_complete_frame_emitted(self):
        r = SseReframer()
        r.feed(self._text_delta_frame("hello"))
        frames = r.drain_complete()
        assert len(frames) == 1
        assert frames[0].data_obj["delta"]["text"] == "hello"

    def test_mid_frame_byte_split(self):
        """httpx splits a frame at an arbitrary byte boundary."""
        frame_bytes = self._text_delta_frame("hello world this is long enough")
        cut = len(frame_bytes) // 2
        r = SseReframer()
        r.feed(frame_bytes[:cut])
        assert r.drain_complete() == []
        r.feed(frame_bytes[cut:])
        frames = r.drain_complete()
        assert len(frames) == 1
        assert frames[0].data_obj["delta"]["text"] == "hello world this is long enough"

    def test_multiple_frames_in_one_feed(self):
        r = SseReframer()
        r.feed(self._text_delta_frame("a") + self._text_delta_frame("b"))
        frames = r.drain_complete()
        assert len(frames) == 2
        assert [f.data_obj["delta"]["text"] for f in frames] == ["a", "b"]

    def test_remaining_on_stream_end(self):
        r = SseReframer()
        r.feed(b"event: partial\ndata: {")
        assert r.drain_complete() == []
        leftover = r.remaining()
        assert b"partial" in leftover

    def test_empty_stream(self):
        r = SseReframer()
        assert r.drain_complete() == []
        assert r.remaining() == b""


# --------------------------------------------------------------------------- #
# Delta extraction — text_delta, thinking_delta, input_json_delta (inv 1/7)
# --------------------------------------------------------------------------- #
class TestDeltaExtraction:
    def test_text_delta_extracted(self):
        from lexvault.streaming.reframer import Frame

        frame = Frame(
            raw=b"",
            event="content_block_delta",
            data_raw="{}",
            data_obj={"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}},
        )
        slots = extract_frame_text(frame)
        assert slots == [("delta.text", "hi")]

    def test_thinking_delta_extracted(self):
        """Invariant 1/7: extended thinking CAN echo masked codenames — restored."""
        from lexvault.streaming.reframer import Frame

        frame = Frame(
            raw=b"",
            event="content_block_delta",
            data_raw="{}",
            data_obj={
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "let me check [LEX-AAAAAAAA]"},
            },
        )
        slots = extract_frame_text(frame)
        assert slots == [("delta.thinking", "let me check [LEX-AAAAAAAA]")]

    def test_input_json_delta_extracted(self):
        from lexvault.streaming.reframer import Frame

        frame = Frame(
            raw=b"",
            event="content_block_delta",
            data_raw="{}",
            data_obj={
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"query":"[LEX-BBBBBBBB]"}'},
            },
        )
        slots = extract_frame_text(frame)
        assert slots == [("delta.partial_json", '{"query":"[LEX-BBBBBBBB]"}')]

    def test_signature_delta_not_extracted(self):
        """signature_delta carries no restorable text — pass through untouched."""
        from lexvault.streaming.reframer import Frame

        frame = Frame(
            raw=b"",
            event="content_block_delta",
            data_raw="{}",
            data_obj={
                "type": "content_block_delta",
                "delta": {"type": "signature_delta", "signature": "WyQ9cA=="},
            },
        )
        assert extract_frame_text(frame) == []

    def test_restore_frame_text_writes_back(self):
        from lexvault.streaming.reframer import Frame

        raw = (
            "event: content_block_delta\ndata: "
            + json.dumps(
                {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": "[LEX-AAAAAAAA]"},
                }
            )
            + "\n\n"
        ).encode()
        frame = Frame(
            raw=raw,
            event="content_block_delta",
            data_raw="{}",
            data_obj=json.loads(raw.split(b"data: ", 1)[1].strip()),
        )
        out = restore_frame_text(frame, {"delta.text": "Project Titan"})
        assert b"Project Titan" in out
        assert b"[LEX-AAAAAAAA]" not in out


# --------------------------------------------------------------------------- #
# streaming_restore dispatcher — bytes + objects regimes (inv 6, 1, 11)
# --------------------------------------------------------------------------- #
class TestStreamingRestore:
    async def test_bytes_regime_restores_text_delta(self, tmp_path):
        """Raw SSE bytes with a placeholder in a text_delta are restored."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-AAAAAAAA]", "Project Titan", request_id="r1")

        payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "see [LEX-AAAAAAAA] now"},
        }
        chunk = f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()

        async def upstream():
            yield chunk

        out_chunks = []
        async for c in streaming_restore(
            upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
        ):
            out_chunks.append(c)
        full = b"".join(out_chunks)
        assert b"Project Titan" in full
        assert b"[LEX-AAAAAAAA]" not in full
        await vault.close()

    async def test_bytes_regime_restores_thinking_delta(self, tmp_path):
        """A codename in a thinking_delta is restored (invariant 1/7 edge)."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-CCCCCCCC]", "Project Mercury", request_id="r1")

        payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "planning around [LEX-CCCCCCCC]"},
        }
        chunk = f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()

        async def upstream():
            yield chunk

        out = b"".join(
            [
                c
                async for c in streaming_restore(
                    upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
                )
            ]
        )
        assert b"Project Mercury" in out
        assert b"[LEX-CCCCCCCC]" not in out
        await vault.close()

    async def test_bytes_split_across_frames(self, tmp_path):
        """A placeholder whose frame is split across two raw byte yields restores."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-DDDDDDDD]", "Codename Z", request_id="r1")

        payload = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "[LEX-DDDDDDDD]"},
        }
        full = f"event: content_block_delta\ndata: {json.dumps(payload)}\n\n".encode()
        cut = len(full) // 2

        async def upstream():
            yield full[:cut]
            yield full[cut:]

        out = b"".join(
            [
                c
                async for c in streaming_restore(
                    upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
                )
            ]
        )
        assert b"Codename Z" in out
        assert b"[LEX-DDDDDDDD]" not in out
        await vault.close()

    async def test_empty_stream(self, tmp_path):
        vault = MappingVault(tmp_path / "s.db")

        async def upstream():
            return
            yield  # pragma: no cover

        out = [
            c
            async for c in streaming_restore(
                upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
            )
        ]
        assert out == []
        await vault.close()

    async def test_stream_ending_mid_placeholder_fails_closed(self, tmp_path):
        """A stream that ends holding a partial placeholder opener → StreamingError."""
        vault = MappingVault(tmp_path / "s.db")

        async def upstream():
            # Plain text that ends mid-placeholder-opener and never closes.
            yield (
                b"event: content_block_delta\ndata: "
                + json.dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "ending [LEX-AAA"},
                    }
                ).encode()
                + b"\n\n"
            )

        with pytest.raises(StreamingError):
            async for _ in streaming_restore(
                upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
            ):
                pass
        await vault.close()

    async def test_parsed_object_regime_restores_delta_content(self, tmp_path):
        """OpenAI ModelResponseStream-style chunks (objects) are restored."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-EEEEEEEE]", "Project Titan", request_id="r1")

        class Delta:
            def __init__(self, content: str) -> None:
                self.content = content

        class Choice:
            def __init__(self, content: str) -> None:
                self.delta = Delta(content)

        class Chunk:
            def __init__(self, content: str) -> None:
                self.choices = [Choice(content)]

        async def upstream():
            yield Chunk("see [LEX-EEEEEEEE] now")

        out = [
            c
            async for c in streaming_restore(
                upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
            )
        ]
        # The restored text may be re-chunked (buffering holds the trailing
        # window); assert on the COMBINED content, not a single chunk.
        combined = "".join(c.choices[0].delta.content for c in out if c.choices[0].delta.content)
        assert "Project Titan" in combined
        assert "[LEX-" not in combined
        await vault.close()


# --------------------------------------------------------------------------- #
# error-frame helpers
# --------------------------------------------------------------------------- #
class TestErrorFrames:
    def test_openai_error_chunk_is_valid_json(self):
        chunk = openai_error_chunk("boom")
        assert chunk.startswith(b"data: ")
        payload = json.loads(chunk[len(b"data: ") :].strip())
        assert payload["error"]["message"] == "boom"

    def test_anthropic_error_event_has_event_line(self):
        frame = anthropic_error_event("boom")
        assert b"event: error" in frame
        payload = json.loads(frame.split(b"data: ", 1)[1].strip())
        assert payload["error"]["message"] == "boom"
