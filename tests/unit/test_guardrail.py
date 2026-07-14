"""Unit tests for the LiteLLM guardrail hooks (no proxy needed).

Instantiates ``LexVaultGuardrail`` and awaits the four hooks directly with mock
``data``/``response`` payloads, per the plan's two-tier testing (in-repo
guardrail-test pattern). Pins IMPLICIT_SPEC invariants:
  1 (round-trip), 3 (vault as cross-hook store), 4 (no apply_guardrail),
  7 (OpenAI + Anthropic parity), 8 (tool-call coverage), 11 (fail-closed),
  19 (logging re-mask), 20 (non-text no-op), 21 (scoping), 24 (per-request config).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lexvault.guardrail import LexVaultGuardrail


@pytest.fixture
def guardrail(tmp_path: Path) -> LexVaultGuardrail:
    """A guardrail with a small dictionary + regex, vault in a temp dir."""
    dict_path = tmp_path / "dict.yaml"
    dict_path.write_text(
        "terms:\n  - {term: Project Titan, type: codename}\n"
        "  - {term: customer_database, type: schema}\n"
        "regex_terms:\n  - {name: Employee ID, pattern: 'EMP-\\\\d{4,6}', type: id}\n",
        encoding="utf-8",
    )
    return LexVaultGuardrail(
        dictionary_path=str(dict_path),
        org_key="test-org-key",
        vault_path=str(tmp_path / "vault.db"),
    )


# --------------------------------------------------------------------------- #
# invariant 4: NO apply_guardrail
# --------------------------------------------------------------------------- #
class TestNoApplyGuardrail:
    def test_apply_guardrail_not_in_dict(self):
        assert "apply_guardrail" not in LexVaultGuardrail.__dict__

    def test_individual_hooks_overridden(self):
        for hook in (
            "async_pre_call_hook",
            "async_post_call_success_hook",
            "async_post_call_streaming_iterator_hook",
            "async_logging_hook",
        ):
            assert hook in LexVaultGuardrail.__dict__, f"{hook} must be overridden"


# --------------------------------------------------------------------------- #
# invariant 20/21: non-text call types no-op; tool defs not masked
# --------------------------------------------------------------------------- #
class TestNoOpAndScoping:
    async def test_non_text_call_type_noop(self, guardrail):
        data = {"messages": [{"role": "user", "content": "Project Titan"}]}
        result = await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "embedding")
        assert result is None
        # Original term untouched.
        assert data["messages"][0]["content"] == "Project Titan"

    async def test_tool_definitions_not_masked(self, guardrail):
        """Invariant 21: data['tools'] schemas are left verbatim."""
        data = {
            "messages": [{"role": "user", "content": "use Project Titan"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "query", "parameters": {"project": "Project Titan"}},
                }
            ],
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        # The tool schema still contains the original term verbatim.
        assert data["tools"][0]["function"]["parameters"]["project"] == "Project Titan"
        # But the message content was masked.
        assert "Project Titan" not in data["messages"][0]["content"]


# --------------------------------------------------------------------------- #
# invariant 1/7: OpenAI + Anthropic round-trip through the hooks
# --------------------------------------------------------------------------- #
class TestOpenAIRoundTrip:
    async def test_mask_openai_request(self, guardrail):
        data = {"messages": [{"role": "user", "content": "Tell me about Project Titan"}]}
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        content = data["messages"][0]["content"]
        assert "Project Titan" not in content
        assert "[LEX-" in content

    async def test_mask_system_role_gated(self, tmp_path):
        dict_path = tmp_path / "d.yaml"
        dict_path.write_text("terms:\n  - {term: Project Titan}\n", encoding="utf-8")
        g_mask = LexVaultGuardrail(
            dictionary_path=str(dict_path), org_key="k", vault_path=str(tmp_path / "v.db")
        )
        g_nomask = LexVaultGuardrail(
            dictionary_path=str(dict_path),
            org_key="k",
            vault_path=str(tmp_path / "v2.db"),
            mask_system_role=False,
        )

        data1 = {
            "messages": [
                {"role": "system", "content": "You know Project Titan"},
                {"role": "user", "content": "hi"},
            ]
        }
        await g_mask.async_pre_call_hook(MagicMock(), MagicMock(), data1, "completion")
        assert "Project Titan" not in data1["messages"][0]["content"]

        data2 = {
            "messages": [
                {"role": "system", "content": "You know Project Titan"},
                {"role": "user", "content": "hi"},
            ]
        }
        await g_nomask.async_pre_call_hook(MagicMock(), MagicMock(), data2, "completion")
        # system role NOT masked.
        assert data2["messages"][0]["content"] == "You know Project Titan"

    async def test_restore_openai_response(self, guardrail):
        # Simulate the model echoing a placeholder back.
        message = SimpleNamespace(content="The answer is [LEX-AAAAAAAA]", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], model="gpt-4o")
        # Seed the vault mapping via a pre_call mask.
        data = {
            "messages": [{"role": "user", "content": "Project Titan"}],
            "litellm_call_id": "req-1",
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        masked_content = data["messages"][0]["content"]
        # Now put that placeholder in the response and restore.
        message.content = f"The answer is {masked_content.replace('Project Titan', '')}"
        # Actually: simpler — derive the placeholder the same way and echo it.
        from lexvault.engine import derive_placeholder

        ph = derive_placeholder("Project Titan", "test-org-key", "default")
        message.content = f"The answer is {ph}"

        await guardrail.async_post_call_success_hook(data, MagicMock(), response)
        assert "Project Titan" in message.content
        assert "[LEX-" not in message.content


class TestAnthropicRoundTrip:
    async def test_mask_anthropic_request_content_blocks(self, guardrail):
        data = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Discuss Project Titan"}]}
            ]
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "anthropic_messages")
        block = data["messages"][0]["content"][0]
        assert "Project Titan" not in block["text"]
        assert "[LEX-" in block["text"]

    async def test_mask_anthropic_system_string(self, guardrail):
        data = {
            "system": "You operate on Project Titan",
            "messages": [{"role": "user", "content": "hi"}],
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "anthropic_messages")
        assert "Project Titan" not in data["system"]

    async def test_restore_anthropic_tool_use_input(self, guardrail):
        """Invariant 8: tool_use.input dict values masked on request and restored on response."""
        # Request masks the tool_use input.
        data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "q",
                            "input": {"project": "Project Titan"},
                        }
                    ],
                }
            ],
            "litellm_call_id": "req-a",
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "anthropic_messages")
        assert "Project Titan" not in data["messages"][0]["content"][0]["input"]["project"]

        # Response carries the placeholder in a tool_use input — restore it.
        from lexvault.engine import derive_placeholder

        ph = derive_placeholder("Project Titan", "test-org-key", "default")
        resp_block = SimpleNamespace(type="tool_use", id="t2", name="q", input={"project": ph})
        response = SimpleNamespace(content=[resp_block], model="claude-3")
        await guardrail.async_post_call_success_hook(data, MagicMock(), response)
        assert resp_block.input["project"] == "Project Titan"


# --------------------------------------------------------------------------- #
# invariant 8: OpenAI tool_calls arguments
# --------------------------------------------------------------------------- #
class TestToolCalls:
    async def test_mask_openai_tool_call_arguments_in_history(self, guardrail):
        data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "t1",
                            "type": "function",
                            "function": {"name": "q", "arguments": '{"project":"Project Titan"}'},
                        }
                    ],
                }
            ],
            "litellm_call_id": "req-t",
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        args = data["messages"][0]["tool_calls"][0]["function"]["arguments"]
        assert "Project Titan" not in args
        assert "[LEX-" in args


# --------------------------------------------------------------------------- #
# invariant 24: per-request config (metadata both locations)
# --------------------------------------------------------------------------- #
class TestPerRequestConfig:
    async def test_per_request_scope_via_metadata_chat_completions(self, tmp_path):
        """Per-request scope override on /v1/chat/completions (uses data['metadata'])."""
        dict_path = tmp_path / "d.yaml"
        dict_path.write_text("terms:\n  - {term: Project Titan}\n", encoding="utf-8")
        g = LexVaultGuardrail(
            dictionary_path=str(dict_path), org_key="k", vault_path=str(tmp_path / "v.db")
        )

        data = {
            "messages": [{"role": "user", "content": "Project Titan"}],
            "metadata": {"requester_metadata": {"lexvault_scope": "custom-scope"}},
            "litellm_call_id": "req-s",
        }
        await g.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        # The mask happened under custom-scope; the vault row exists under that scope.
        from lexvault.engine import derive_placeholder

        ph = derive_placeholder("Project Titan", "k", "custom-scope")
        assert ph in data["messages"][0]["content"]

    async def test_per_request_scope_via_litellm_metadata_messages(self, tmp_path):
        """Per-request scope on /v1/messages (uses data['litellm_metadata'])."""
        dict_path = tmp_path / "d.yaml"
        dict_path.write_text("terms:\n  - {term: Project Titan}\n", encoding="utf-8")
        g = LexVaultGuardrail(
            dictionary_path=str(dict_path), org_key="k", vault_path=str(tmp_path / "v.db")
        )

        data = {
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Project Titan"}]}],
            "litellm_metadata": {"requester_metadata": {"lexvault_scope": "msg-scope"}},
            "litellm_call_id": "req-m",
        }
        await g.async_pre_call_hook(MagicMock(), MagicMock(), data, "anthropic_messages")
        from lexvault.engine import derive_placeholder

        ph = derive_placeholder("Project Titan", "k", "msg-scope")
        assert ph in data["messages"][0]["content"][0]["text"]


# --------------------------------------------------------------------------- #
# invariant 3: cross-hook consistency (same term → same placeholder)
# --------------------------------------------------------------------------- #
class TestCrossHookConsistency:
    async def test_same_term_same_placeholder_request_and_response(self, guardrail):
        """Mask on pre_call and restore on post_call share the vault mapping."""
        data = {
            "messages": [{"role": "user", "content": "Project Titan"}],
            "litellm_call_id": "req-x",
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        masked = data["messages"][0]["content"]
        assert "[LEX-" in masked

        # Echo the masked content in a response; restore it.
        message = SimpleNamespace(content=masked, tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], model="gpt-4o")
        await guardrail.async_post_call_success_hook(data, MagicMock(), response)
        assert message.content == "Project Titan"


# --------------------------------------------------------------------------- #
# invariant 11: fail-closed
# --------------------------------------------------------------------------- #
class TestFailClosed:
    async def test_mask_error_blocks_when_fail_closed(self, guardrail):
        """A vault error during mask raises (no original reaches upstream)."""
        from litellm.exceptions import ModifyResponseException

        guardrail._vault = MagicMock()
        guardrail._vault.assign = MagicMock(side_effect=RuntimeError("vault down"))
        data = {"messages": [{"role": "user", "content": "Project Titan"}], "litellm_call_id": "r"}
        # fail-closed raises (HTTPException under fastapi, RuntimeError otherwise).
        with pytest.raises((RuntimeError, ModifyResponseException)):
            await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")

    async def test_restore_error_sanitizes_when_fail_closed(self, guardrail):
        """A vault error during restore raises ModifyResponseException (sanitized).

        The exception's own payload MUST carry neither the placeholder nor the
        original — that is the no-leak requirement (invariant 11), and merely
        asserting `pytest.raises` would not catch a leak via the exception's
        request_data/original_response fields.
        """
        from litellm.exceptions import ModifyResponseException

        guardrail._vault = MagicMock()
        guardrail._vault._lookup_sync = MagicMock(side_effect=RuntimeError("vault down"))
        message = SimpleNamespace(content="[LEX-AAAAAAAA]", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], model="gpt-4o")
        with pytest.raises(ModifyResponseException) as exc_info:
            await guardrail.async_post_call_success_hook({}, MagicMock(), response)
        # The exception payload must not echo the placeholder or any original.
        rendered = repr(exc_info.value)
        assert "[LEX-" not in rendered
        assert "Project Titan" not in rendered
        assert "[LEX-AAAAAAAA]" not in exc_info.value.message

    async def test_fail_open_returns_data_on_mask_error(self, tmp_path):
        dict_path = tmp_path / "d.yaml"
        dict_path.write_text("terms:\n  - {term: Project Titan}\n", encoding="utf-8")
        g = LexVaultGuardrail(
            dictionary_path=str(dict_path),
            org_key="k",
            vault_path=str(tmp_path / "v.db"),
            fail_open=True,
        )
        g._vault = MagicMock()
        g._vault.assign = MagicMock(side_effect=RuntimeError("vault down"))
        data = {"messages": [{"role": "user", "content": "Project Titan"}], "litellm_call_id": "r"}
        # With fail_open=True, the hook returns data (unmasked) rather than raising.
        result = await g.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        assert result is data


# --------------------------------------------------------------------------- #
# invariant 19: logging re-mask
# --------------------------------------------------------------------------- #
class TestLoggingRemask:
    async def test_logging_hook_remasks_response(self, guardrail):
        """The logging hook re-masks the result payload (defense-in-depth)."""
        # Seed a mapping.
        data = {"messages": [{"role": "user", "content": "Project Titan"}], "litellm_call_id": "r1"}
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")

        # A result that (erroneously) contains an original term — logging must mask it.
        message = SimpleNamespace(content="Project Titan is secret", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        result = SimpleNamespace(choices=[choice], model="gpt-4o")

        kwargs, returned = await guardrail.async_logging_hook(
            {"litellm_call_id": "r1"}, result, "completion"
        )
        assert "Project Titan" not in message.content
        assert "[LEX-" in message.content


# --------------------------------------------------------------------------- #
# invariant 3 edge: missing litellm_call_id is generated (not a crash)
# --------------------------------------------------------------------------- #
class TestMissingCallId:
    async def test_mask_and_restore_without_litellm_call_id(self, guardrail):
        """Invariant 3 edge: a request with NO litellm_call_id still round-trips.

        LiteLLM sets litellm_call_id before pre_call under the proxy, so the only
        way to reach pre_call without one is header spoofing — but the guardrail
        must not crash; it generates a uuid4 (guardrail.py:173) and the
        vault-keyed restore still resolves by *placeholder string* (not by id),
        so round-trip fidelity holds.
        """
        data = {"messages": [{"role": "user", "content": "Tell me about Project Titan"}]}
        # NO litellm_call_id key present.
        assert "litellm_call_id" not in data
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "completion")
        masked = data["messages"][0]["content"]
        assert "Project Titan" not in masked
        assert "[LEX-" in masked

        # Restore resolves by placeholder string, so the missing id is harmless.
        from lexvault.engine import derive_placeholder

        ph = derive_placeholder("Project Titan", "test-org-key", "default")
        message = SimpleNamespace(content=f"The answer is {ph}", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice], model="gpt-4o")
        await guardrail.async_post_call_success_hook(data, MagicMock(), response)
        assert "Project Titan" in message.content
        assert "[LEX-" not in message.content
