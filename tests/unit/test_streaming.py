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

    # -- CS-E1: opener-prefix-aware hold-back (the reproduced blocker) -------- #

    def test_opener_prefix_cut_does_not_leak(self):
        """CS-E1: a cut landing INSIDE the opener prefix must not leak the placeholder.

        Feeding 'x'*10 + '[LEX-AAAAAAAA]' + 'z'*6 with window 19 puts the cut at
        offset 11 — the ready slice ends with '[' (a prefix of the opener '[LEX-').
        Before the fix no hold-back fired and the full placeholder was emitted
        verbatim. This is the exact RC4 split-placeholder leak the product exists
        to prevent.
        """
        placeholder = "[LEX-AAAAAAAA]"

        def lookup(ph: str) -> str | None:
            return "Codename" if ph == placeholder else None

        buf = PlaceholderBuffer(19, NS)
        buf.feed("x" * 10 + placeholder + "z" * 6)  # len 30; cut = 30 - 19 = 11
        out = buf.drain_restored_with_vault(lookup)
        out += buf.flush_restored_with_vault(lookup)[0]
        assert "Codename" in out
        assert placeholder not in out
        assert "[LEX-" not in out  # no partial opener leaked either

    @pytest.mark.parametrize("offset", range(len("[LEX-AAAAAAAA]") + 1))
    def test_placeholder_split_at_every_feed_offset(self, offset):
        """CS-E1: a placeholder split across two feeds at ANY internal offset
        (including every offset inside the opener '[LEX-') restores whole.

        Exhaustive over 0..len(placeholder): the cut sweeps through every byte
        of the placeholder across the two feed boundaries.
        """
        placeholder = "[LEX-AAAAAAAA]"

        def lookup(ph: str) -> str | None:
            return "Codename" if ph == placeholder else None

        prefix = "padding!!"  # 9 chars before the placeholder
        suffix = "trailing!!"  # 10 chars after
        text = prefix + placeholder + suffix
        split = len(prefix) + offset  # cut at offset within / just past placeholder

        buf = PlaceholderBuffer(WINDOW, NS)
        buf.feed(text[:split])
        out = buf.drain_restored_with_vault(lookup)
        buf.feed(text[split:])
        out += buf.drain_restored_with_vault(lookup)
        out += buf.flush_restored_with_vault(lookup)[0]
        assert "Codename" in out
        assert placeholder not in out
        assert "[LEX-" not in out

    def test_flush_partial_when_tail_is_opener_prefix(self):
        """CS-E1: a tail ending in a proper prefix of the opener (e.g. '[LEX')
        is a potential partial placeholder — flush must signal partial_in_namespace
        (fail-closed), not emit it as plain text."""
        buf = PlaceholderBuffer(WINDOW, NS)
        buf.feed("ending [LEX")  # '[LEX' is a proper prefix of opener '[LEX-'
        _remaining, partial = buf.flush_restored_with_vault(_noop_lookup)
        assert partial is True


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

    async def test_bytes_thinking_delta_stays_thinking_type(self, tmp_path):
        """CS-E3: a codename in a thinking_delta restores but the frame stays
        ``thinking_delta`` (not converted to a ``text_delta``). Type fidelity."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-CCCCCCCC]", "Project Mercury", request_id="r1")
        payload = {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "thinking_delta", "thinking": "plan [LEX-CCCCCCCC] done"},
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
        # The restored frame must still be a thinking_delta carrying the original.
        assert b'"thinking_delta"' in out
        assert b'"text_delta"' not in out
        assert b"Project Mercury" in out
        assert b"[LEX-CCCCCCCC]" not in out
        await vault.close()

    async def test_bytes_input_json_delta_restored_in_place(self, tmp_path):
        """CS-E3: tool-call partial_json (input_json_delta) restores in place as
        input_json_delta, not converted to text. Tool args stay tool args."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-BBBBBBBB]", "Project Mercury", request_id="r1")
        payload = {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"q":"[LEX-BBBBBBBB]"}'},
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
        assert b'"input_json_delta"' in out
        assert b'"partial_json"' in out
        assert b'"text_delta"' not in out
        assert b"Project Mercury" in out
        assert b"[LEX-BBBBBBBB]" not in out
        await vault.close()

    async def test_bytes_empty_text_content_block_start_survives(self, tmp_path):
        """CS-E3: a content_block_start with empty text:"" must pass through
        (not be dropped as non-restorable). Every normal text stream starts
        with one; dropping it corrupts the stream."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-AAAAAAAA]", "Project Titan", request_id="r1")
        start = (
            b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
            b'"content_block":{"type":"text","text":""}}\n\n'
        )
        delta = f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': '[LEX-AAAAAAAA]'}})}\n\n".encode()
        stop = b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'

        async def upstream():
            yield start + delta + stop

        out = b"".join(
            [
                c
                async for c in streaming_restore(
                    upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
                )
            ]
        )
        # The empty-text content_block_start MUST appear in the restored stream.
        assert b"content_block_start" in out
        assert b'"text": ""' in out or b'"text":""' in out
        assert b"content_block_stop" in out
        assert b"Project Titan" in out
        assert b"[LEX-AAAAAAAA]" not in out
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

    async def test_stream_ending_mid_placeholder_yields_no_partial(self, tmp_path):
        """CS-E2: a dangling partial placeholder at stream end must NOT appear in
        the yielded output before the error frame — the tail is suppressed so no
        partial placeholder can reach the client. Asserts on collected chunks."""
        vault = MappingVault(tmp_path / "s.db")

        async def upstream():
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

        out = []
        with pytest.raises(StreamingError):
            async for c in streaming_restore(
                upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
            ):
                out.append(c)
        # The dangling partial placeholder '[LEX-AAA' must not be in any frame
        # yielded to the client before the error frame.
        joined = b"".join(out)
        assert b"[LEX-AAA" not in joined
        assert b"[LEX-" not in joined
        await vault.close()

    async def test_vault_error_mid_stream_raises_streaming_error(self, tmp_path):
        """CS-E5: a VaultError raised by the vault during restore must surface as
        StreamingError (routing through the error-frame path), not propagate raw
        as an unhandled exception that aborts the stream without a sanitized frame."""

        class _ExplodingVault:
            """Stand-in for a MappingVault whose _lookup_sync raises mid-stream."""

            def _lookup_sync(self, _placeholder: str) -> str | None:  # noqa: SLF001
                raise RuntimeError("vault unavailable")

        async def upstream():
            # A frame with a real placeholder; restore calls _lookup_sync → raise.
            yield (
                b"event: content_block_delta\ndata: "
                + json.dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "see [LEX-AAAAAAAA] now"},
                    }
                ).encode()
                + b"\n\n"
            )

        with pytest.raises(StreamingError):
            async for _ in streaming_restore(
                upstream(),
                vault=_ExplodingVault(),  # type: ignore[arg-type]
                placeholder_namespace_re=NS,
                max_placeholder_len=WINDOW,
            ):
                pass

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

    async def test_parsed_regime_preserves_chunk_metadata_and_identity(self, tmp_path):
        """CS-E6: restore MUTATES the original ModelResponseStream chunks in place
        (preserving id/model/created/finish_reason and chunk identity), instead of
        synthesizing SimpleNamespace chunks that drop metadata. Multiple chunks
        are split so a placeholder straddles the chunk boundary."""
        vault = MappingVault(tmp_path / "s.db")
        await vault.assign("default", "[LEX-EEEEEEEE]", "Project Titan", request_id="r1")

        class Delta:
            def __init__(self, content: str) -> None:
                self.content = content

        class Choice:
            def __init__(self, content: str, finish_reason=None) -> None:
                self.delta = Delta(content)
                self.finish_reason = finish_reason

        class Chunk:
            def __init__(self, content: str, *, cid: str, model: str, fr=None) -> None:
                self.id = cid
                self.model = model
                self.created = 1700000000
                self.object = "chat.completion.chunk"
                self.choices = [Choice(content, fr)]

        placeholder = "[LEX-EEEEEEEE]"
        c1 = Chunk("see " + placeholder[:4], cid="ch1", model="gpt-4o")
        c2 = Chunk(placeholder[4:] + " now", cid="ch2", model="gpt-4o", fr="stop")
        originals = [c1, c2]

        async def upstream():
            yield c1
            yield c2

        out = [
            c
            async for c in streaming_restore(
                upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
            )
        ]
        # Combined restored content is correct.
        combined = "".join(getattr(c.choices[0].delta, "content", "") for c in out if c.choices)
        assert "Project Titan" in combined
        assert placeholder not in combined

        # CS-E6: every emitted content chunk is one of the ORIGINAL chunk objects
        # (mutated in place), not a synthesized SimpleNamespace — so the client's
        # proxy serialization keeps id/model/created/object/finish_reason.
        emitted_originals = [
            c for c in out if getattr(c, "object", None) == "chat.completion.chunk"
        ]
        assert emitted_originals, "expected original ModelResponseStream chunks to be emitted"
        for c in emitted_originals:
            assert c in originals
            assert c.model == "gpt-4o"
            assert c.created == 1700000000
        # The final chunk's finish_reason is preserved.
        assert any(c.choices[0].finish_reason == "stop" for c in emitted_originals if c.choices)
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
