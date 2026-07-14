"""Integration tests: end-to-end mask→LLM→restore through the guardrail.

Tier-2 integration per the plan's two-tier testing. Rather than boot a full
LiteLLM proxy + fake HTTP backend (heavy, flaky in CI), we use LiteLLM's own
``mock_response`` mechanism to produce realistic ``ModelResponse`` /
``ModelResponseStream`` objects and drive them through the guardrail's hooks.
This validates the full adapter → engine → restore path on real litellm shapes.

Pins IMPLICIT_SPEC invariants:
  1 (round-trip), 3 (cross-hook via vault), 7 (OpenAI parity), 8 (tool calls),
  1 (streaming round-trip including split placeholders).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from litellm.exceptions import ModifyResponseException

from lexvault.engine import derive_placeholder
from lexvault.guardrail import LexVaultGuardrail

pytestmark = pytest.mark.asyncio


@pytest.fixture
def gr(tmp_path: Path) -> LexVaultGuardrail:
    dict_path = tmp_path / "dict.yaml"
    dict_path.write_text(
        "terms:\n  - {term: Project Titan, type: codename}\n"
        "  - {term: customer_database, type: schema}\n",
        encoding="utf-8",
    )
    return LexVaultGuardrail(
        dictionary_path=str(dict_path),
        org_key="integration-key",
        vault_path=str(tmp_path / "vault.db"),
    )


# --------------------------------------------------------------------------- #
# Non-streaming OpenAI: mock_response round-trip
# --------------------------------------------------------------------------- #
class TestOpenAINonStreamRoundTrip:
    async def test_mock_response_round_trip(self, gr):
        """Mask a request, get a mock response echoing the placeholder, restore it."""
        import litellm

        # 1. Mask the request: simulate pre_call on a chat-completions body.
        data = {
            "messages": [{"role": "user", "content": "What is Project Titan?"}],
            "litellm_call_id": "req-1",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        masked = data["messages"][0]["content"]
        assert "Project Titan" not in masked
        placeholder = [p for p in [masked] if "[LEX-" in p]
        assert placeholder

        # 2. Get a real ModelResponse via litellm mock_response, echoing the placeholder.
        ph = derive_placeholder("Project Titan", "integration-key", "default")
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            mock_response=f"The codename is {ph}",
        )
        # 3. Restore via post_call.
        await gr.async_post_call_success_hook(data, MagicMock(), response)
        restored = response.choices[0].message.content
        assert "Project Titan" in restored
        assert "[LEX-" not in restored

    async def test_multiple_terms_round_trip(self, gr):
        import litellm

        data = {
            "messages": [{"role": "user", "content": "Query customer_database for Project Titan"}],
            "litellm_call_id": "req-2",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        assert "[LEX-" in data["messages"][0]["content"]

        ph_a = derive_placeholder("customer_database", "integration-key", "default")
        ph_b = derive_placeholder("Project Titan", "integration-key", "default")
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            mock_response=f"Use {ph_a} with {ph_b}",
        )
        await gr.async_post_call_success_hook(data, MagicMock(), response)
        content = response.choices[0].message.content
        assert "customer_database" in content
        assert "Project Titan" in content
        assert "[LEX-" not in content

    async def test_no_placeholders_in_response_passes_through(self, gr):
        """A response with no placeholders is returned unchanged."""
        import litellm

        data = {"messages": [{"role": "user", "content": "hi"}], "litellm_call_id": "req-3"}
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello there",
        )
        original = response.choices[0].message.content
        await gr.async_post_call_success_hook(data, MagicMock(), response)
        assert response.choices[0].message.content == original


# --------------------------------------------------------------------------- #
# Fail-closed: restore miss sanitizes (no leak)
# --------------------------------------------------------------------------- #
class TestFailClosedIntegration:
    async def test_restore_miss_sanitizes(self, gr):
        """A response containing a placeholder NOT in the vault → ModifyResponseException.

        Fail-closed (invariant 11): an unresolved placeholder in the response is a
        potential leak. We assert not only that the exception is raised, but that
        the exception's own payload (message/request_data) carries NEITHER the
        placeholder NOR any original — a bare `pytest.raises` would not catch a
        leak through the exception attributes.
        """
        import litellm

        data = {"messages": [{"role": "user", "content": "hi"}], "litellm_call_id": "req-f"}
        # A placeholder the vault has no mapping for.
        response = litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "x"}],
            mock_response="Leaked [LEX-ZZZZZZZZ] here",
        )
        with pytest.raises(ModifyResponseException) as exc_info:
            await gr.async_post_call_success_hook(data, MagicMock(), response)
        # No-leak on the exception payload itself.
        rendered = repr(exc_info.value)
        assert "[LEX-" not in rendered
        assert "Project Titan" not in rendered
        assert "customer_database" not in rendered
        assert "[LEX-" not in exc_info.value.message

    async def test_no_original_in_request_reaches_masked_form(self, gr):
        """After pre_call, NO original term is anywhere in the masked request body."""
        data = {
            "messages": [
                {"role": "system", "content": "You know Project Titan"},
                {"role": "user", "content": "Tell me about customer_database"},
            ],
            "litellm_call_id": "req-n",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        body_str = str(data)
        assert "Project Titan" not in body_str
        assert "customer_database" not in body_str


# --------------------------------------------------------------------------- #
# Streaming: ModelResponseStream with a placeholder split across chunks
# --------------------------------------------------------------------------- #
class TestStreamingRoundTrip:
    async def test_streaming_placeholder_restored(self, gr):
        """A placeholder in a streaming delta is restored end-to-end."""
        # Seed the mapping.
        seed_data = {
            "messages": [{"role": "user", "content": "Project Titan"}],
            "litellm_call_id": "req-s",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), seed_data, "completion")

        ph = derive_placeholder("Project Titan", "integration-key", "default")

        # Build fake ModelResponseStream chunks with the placeholder in delta.content.
        from types import SimpleNamespace

        def chunk(content: str):
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
            )

        async def upstream():
            yield chunk("The codename is ")
            yield chunk(ph)

        out = []
        async for c in gr.async_post_call_streaming_iterator_hook(
            MagicMock(), upstream(), seed_data
        ):
            out.append(c)
        restored = "".join(c.choices[0].delta.content for c in out)
        assert "Project Titan" in restored
        assert "[LEX-" not in restored

    async def test_streaming_placeholder_split_across_chunks(self, gr):
        """A placeholder split across 3 streaming chunks restores whole."""
        seed_data = {
            "messages": [{"role": "user", "content": "Project Titan"}],
            "litellm_call_id": "req-split",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), seed_data, "completion")

        ph = derive_placeholder("Project Titan", "integration-key", "default")
        # Split the placeholder into 3 pieces.
        third = len(ph) // 3
        parts = [ph[:third], ph[third : 2 * third], ph[2 * third :]]

        from types import SimpleNamespace

        def chunk(content: str):
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
            )

        async def upstream():
            yield chunk("Answer: ")
            for p in parts:
                yield chunk(p)
            yield chunk(" done")

        out = []
        async for c in gr.async_post_call_streaming_iterator_hook(
            MagicMock(), upstream(), seed_data
        ):
            out.append(c)
        restored = "".join(c.choices[0].delta.content for c in out if c.choices[0].delta.content)
        assert "Project Titan" in restored
        assert "[LEX-" not in restored


# --------------------------------------------------------------------------- #
# Anthropic-native content blocks round-trip
# --------------------------------------------------------------------------- #
class TestAnthropicRoundTrip:
    async def test_anthropic_content_blocks_mask_and_restore(self, gr):
        """Anthropic text content blocks mask on request, restore on response."""
        data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Discuss Project Titan"}]}
            ],
            "litellm_call_id": "req-a",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), data, "anthropic_messages")
        assert "Project Titan" not in data["messages"][0]["content"][0]["text"]

        ph = derive_placeholder("Project Titan", "integration-key", "default")
        # Simulate an AnthropicMessagesResponse with a content block.
        from types import SimpleNamespace

        block = SimpleNamespace(type="text", text=f"The answer is {ph}")
        response = SimpleNamespace(content=[block], model="claude-3")
        await gr.async_post_call_success_hook(data, MagicMock(), response)
        assert "Project Titan" in block.text
        assert "[LEX-" not in block.text

    async def test_anthropic_response_with_both_text_and_tool_use_blocks(self, gr):
        """Invariant 7 edge: a response mixing text + tool_use blocks restores BOTH.

        This is the #22821 RC1/RC4 shape incumbents fail — a model that emits a
        text block referencing the codename AND a tool_use whose input carries
        it. Both must restore.
        """
        from types import SimpleNamespace

        # Seed both terms.
        seed = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Query Project Titan in customer_database"}
                    ],
                }
            ],
            "litellm_call_id": "req-both",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), seed, "anthropic_messages")

        ph_titan = derive_placeholder("Project Titan", "integration-key", "default")
        ph_db = derive_placeholder("customer_database", "integration-key", "default")
        text_block = SimpleNamespace(type="text", text=f"Using {ph_titan} now")
        tool_block = SimpleNamespace(
            type="tool_use", id="t1", name="query", input={"project": ph_titan, "source": ph_db}
        )
        response = SimpleNamespace(content=[text_block, tool_block], model="claude-3")
        await gr.async_post_call_success_hook(seed, MagicMock(), response)

        # Text block restored.
        assert "Project Titan" in text_block.text
        assert "[LEX-" not in text_block.text
        # tool_use input dict restored (both string values).
        assert tool_block.input["project"] == "Project Titan"
        assert tool_block.input["source"] == "customer_database"
        assert "[LEX-" not in str(tool_block.input)


# --------------------------------------------------------------------------- #
# OpenAI output tool_calls restore (invariant 8 — the OUTPUT half)
# --------------------------------------------------------------------------- #
class TestOpenAIOutputToolCalls:
    async def test_openai_output_tool_call_arguments_restored(self, gr):
        """Invariant 8 output half: tool_calls[].function.arguments (JSON string)
        emitted by the model are restored before reaching the client.

        The history-mask test covers the request side; this covers the response
        side, which is the half #31950 documents Presidio leaking.
        """
        from types import SimpleNamespace

        seed = {
            "messages": [{"role": "user", "content": "query Project Titan"}],
            "litellm_call_id": "req-out-tc",
        }
        await gr.async_pre_call_hook(MagicMock(), MagicMock(), seed, "completion")

        ph = derive_placeholder("Project Titan", "integration-key", "default")
        # A ModelResponse whose tool_calls arguments JSON-string carries the placeholder.
        fn = SimpleNamespace(name="query", arguments=f'{{"project":"{ph}"}}')
        tool_call = SimpleNamespace(id="t1", type="function", function=fn)
        message = SimpleNamespace(content="done", tool_calls=[tool_call])
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        response = SimpleNamespace(choices=[choice], model="gpt-4o")

        await gr.async_post_call_success_hook(seed, MagicMock(), response)
        assert tool_call.function.arguments == '{"project":"Project Titan"}'
        assert "[LEX-" not in tool_call.function.arguments


# --------------------------------------------------------------------------- #
# Concurrency: no cross-contamination
# --------------------------------------------------------------------------- #
class TestConcurrency:
    async def test_concurrent_requests_no_cross_contamination(self, tmp_path):
        """100 concurrent mask+restore cycles with DISTINCT terms per request.

        Each request masks a *different* original and restores the echoed
        placeholder back to that SAME original. This is the test that would FAIL
        if the guardrail stored mappings in an instance attribute (the Presidio
        bug): under asyncio.gather one request could restore another's original.
        Using the identical term for all 100 requests (as the original test did)
        makes this tautological — the same placeholder can only ever map to the
        same original, so no cross-contamination is detectable. Here we build a
        100-term dictionary so each request masks a genuinely distinct term to a
        genuinely distinct placeholder, then assert each restores to its OWN
        original and the 100 originals are all distinct.
        """
        import asyncio
        from types import SimpleNamespace

        from lexvault.engine import derive_placeholder
        from lexvault.guardrail import LexVaultGuardrail

        original_terms = [f"Codename-{i:03d}" for i in range(100)]
        dict_lines = "\n".join(f"  - {{term: {t}, type: codename}}" for t in original_terms)
        dict_path = tmp_path / "dict100.yaml"
        dict_path.write_text(f"terms:\n{dict_lines}\n", encoding="utf-8")
        gr = LexVaultGuardrail(
            dictionary_path=str(dict_path),
            org_key="integration-key",
            vault_path=str(tmp_path / "vault100.db"),
        )
        expected_placeholders = {
            t: derive_placeholder(t, "integration-key", "default") for t in original_terms
        }
        # Sanity: 100 distinct terms must yield 100 distinct placeholders.
        assert len(set(expected_placeholders.values())) == 100

        results: dict[int, tuple[str, str]] = {}
        lock = asyncio.Lock()

        async def one(i: int) -> None:
            term = original_terms[i]
            ph = expected_placeholders[term]
            data = {
                "messages": [{"role": "user", "content": term}],
                "litellm_call_id": f"req-c-{i}",
            }
            # MASK.
            await gr.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
            masked_content = data["messages"][0]["content"]
            # The request must NOT carry the original; it MUST carry THIS term's placeholder.
            assert term not in masked_content
            assert ph in masked_content
            # RESTORE: echo this request's placeholder back through a response.
            message = SimpleNamespace(content=f"echo {ph}", tool_calls=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            response = SimpleNamespace(choices=[choice], model="gpt-4o")
            await gr.async_post_call_success_hook(data, MagicMock(), response)
            restored = message.content
            # Cross-contamination check: restored MUST contain THIS request's original.
            assert term in restored, f"req {i}: expected {term!r} in {restored!r}"
            async with lock:
                results[i] = (ph, restored)

        await asyncio.gather(*(one(i) for i in range(100)))

        # Every request restored to its OWN distinct original — no two swapped.
        restored_originals = {results[i][1] for i in range(100)}
        assert len(restored_originals) == 100, (
            "some requests restored to the same original (cross-contamination)"
        )
        # And every distinct placeholder resolved to a distinct original.
        placeholder_to_original = {results[i][0]: results[i][1] for i in range(100)}
        assert len(placeholder_to_original) == 100
