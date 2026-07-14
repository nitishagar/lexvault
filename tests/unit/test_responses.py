"""Tests for the /v1/responses adapter path (CS-C8/E8).

The Testing Strategy claimed coverage of '/v1/responses text' but no test file
existed. These tests pin invariant 1 (round-trip fidelity) and invariant 8
(text coverage) for the Responses API request (input) and response (output)
shapes, plus the streaming ResponsesAPIStreamingResponse restore path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lexvault.guardrail import LexVaultGuardrail
from lexvault.streaming.restore import streaming_restore

NS = r"\[LEX\-[A-Z2-7]+\](?:-\d+)?"
WINDOW = 20


@pytest.fixture
async def guardrail(tmp_path):
    dict_path = tmp_path / "d.yaml"
    dict_path.write_text("terms:\n  - {term: Project Titan}\n", encoding="utf-8")
    g = LexVaultGuardrail(
        dictionary_path=str(dict_path),
        org_key="test-org-key",
        vault_path=str(tmp_path / "v.db"),
    )
    yield g


class TestResponsesRoundTrip:
    async def test_request_input_string_masked(self, guardrail):
        """A /v1/responses request with a plain-string input is masked."""
        data = {"input": "What is Project Titan?", "litellm_call_id": "res-1"}
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "responses")
        assert "Project Titan" not in data["input"]
        assert "[LEX-" in data["input"]

    async def test_request_input_items_list_masked(self, guardrail):
        """A /v1/responses request with input as a list of items is masked."""
        data = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Tell me about Project Titan"}],
                }
            ],
            "litellm_call_id": "res-2",
        }
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), data, "responses")
        text = data["input"][0]["content"][0]["text"]
        assert "Project Titan" not in text
        assert "[LEX-" in text

    async def test_response_output_text_restored(self, guardrail):
        """A /v1/responses response with output_text restores the original."""
        # Seed the mapping via a pre_call mask.
        req = {"input": "Project Titan", "litellm_call_id": "res-3"}
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), req, "responses")
        from lexvault.engine import derive_placeholder

        ph = derive_placeholder("Project Titan", "test-org-key", "default")

        # Response carries the placeholder in output_text.
        response = SimpleNamespace(output_text=f"The answer is {ph}", output=[])
        await guardrail.async_post_call_success_hook({}, MagicMock(), response)
        assert response.output_text == f"The answer is {ph}".replace(ph, "Project Titan")
        assert "Project Titan" in response.output_text
        assert "[LEX-" not in response.output_text

    async def test_response_output_items_restored(self, guardrail):
        """A /v1/responses response with output items (content parts) restores."""
        req = {"input": "Project Titan", "litellm_call_id": "res-4"}
        await guardrail.async_pre_call_hook(MagicMock(), MagicMock(), req, "responses")
        from lexvault.engine import derive_placeholder

        ph = derive_placeholder("Project Titan", "test-org-key", "default")
        # output is a list of items with content parts (dict shape).
        response = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": f"see {ph}"}]}
            ]
        }
        await guardrail.async_post_call_success_hook({}, MagicMock(), response)
        assert response["output"][0]["content"][0]["text"] == "see Project Titan"

    async def test_responses_streaming_restores(self, tmp_path):
        """CS-C8/E8: a ResponsesAPIStreamingResponse-shaped chunk (dict with
        delta.content) restores the placeholder via the parsed-object regime.

        The placeholder may be smaller than the window, so the restored text can
        emerge on a later (flush) chunk — assert on the COMBINED output."""
        from lexvault.vault import MappingVault

        vault = MappingVault(tmp_path / "rs.db")
        await vault.assign("default", "[LEX-EEEEEEEE]", "Project Titan", request_id="r1")

        async def upstream():
            yield {"type": "response.output_text.delta", "delta": {"content": "[LEX-EEEEEEEE]"}}

        out = [
            c
            async for c in streaming_restore(
                upstream(), vault=vault, placeholder_namespace_re=NS, max_placeholder_len=WINDOW
            )
        ]

        def _text(c):
            if isinstance(c, dict):
                return str(c.get("delta", {}).get("content", ""))
            return getattr(
                getattr(c, "choices", [SimpleNamespace(delta=SimpleNamespace(content=""))])[
                    0
                ].delta,
                "content",
                "",
            )

        combined = "".join(_text(c) for c in out)
        assert "Project Titan" in combined
        assert "[LEX-" not in combined
        await vault.close()
