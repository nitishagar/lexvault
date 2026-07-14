"""Unit tests for adapter edge paths (CS-A1 coverage).

Targets the content-parts-list + structured-tool-args + dict-response paths in
the OpenAI adapter, and the structured-args + list-content paths in the
Responses adapter, that the round-trip tests don't fully exercise.
"""

from __future__ import annotations

from types import SimpleNamespace

from lexvault.adapters import openai as openai_adapter
from lexvault.adapters import responses as responses_adapter


class TestOpenAIAdapterEdges:
    def test_request_content_parts_list_slot(self):
        """A message with content as a list of text parts yields a slot per part."""
        data = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        slots = openai_adapter.request_message_slots(data)
        assert len(slots) == 1
        assert slots[0][0]() == "hi"
        slots[0][1]("masked")
        assert data["messages"][0]["content"][0]["text"] == "masked"

    def test_request_non_text_part_ignored(self):
        """A non-text content part (e.g. image_url) yields no slot."""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x"}},
                        {"type": "text", "text": "describe this"},
                    ],
                }
            ]
        }
        slots = openai_adapter.request_message_slots(data)
        assert len(slots) == 1
        assert slots[0][0]() == "describe this"

    def test_request_system_skipped_when_mask_system_false(self):
        data = {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user text"},
            ]
        }
        slots = openai_adapter.request_message_slots(data, mask_system=False)
        # Only the user message — system is skipped.
        assert len(slots) == 1
        assert slots[0][0]() == "user text"

    def test_request_tool_call_arguments_slot(self):
        data = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "t1", "function": {"name": "q", "arguments": '{"a":"b"}'}}
                    ],
                }
            ]
        }
        slots = openai_adapter.request_message_slots(data)
        assert any(s[0]() == '{"a":"b"}' for s in slots)

    def test_response_content_parts_list_slot(self):
        """Response with content as a list of text parts."""
        part = {"type": "text", "text": "hello"}
        message = SimpleNamespace(content=[part], tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        response = SimpleNamespace(choices=[choice])
        slots = openai_adapter.response_slots(response)
        assert len(slots) == 1
        assert slots[0][0]() == "hello"
        slots[0][1]("restored")
        assert part["text"] == "restored"

    def test_response_tool_call_object_args_string(self):
        """A response tool_call whose function.arguments is a string."""
        fn = SimpleNamespace(arguments='{"q": "x"}')
        tc = SimpleNamespace(function=fn)
        message = SimpleNamespace(content=None, tool_calls=[tc])
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        slots = openai_adapter.response_slots(response)
        assert any("q" in s[0]() for s in slots)

    def test_response_tool_call_object_args_dict(self):
        """A response tool_call whose function.arguments is a structured dict."""
        fn = SimpleNamespace(arguments={"q": "structured-val"})
        tc = SimpleNamespace(function=fn)
        message = SimpleNamespace(content=None, tool_calls=[tc])
        choice = SimpleNamespace(message=message)
        response = SimpleNamespace(choices=[choice])
        slots = openai_adapter.response_slots(response)
        assert any(s[0]() == "structured-val" for s in slots)
        # Mutate via setter.
        slots[0][1]("masked-val")
        assert fn.arguments["q"] == "masked-val"

    def test_response_with_no_message_skips_choice(self):
        """A choice with no message attribute yields no slot (defensive skip)."""
        choice = SimpleNamespace(message=None)
        response = SimpleNamespace(choices=[choice])
        slots = openai_adapter.response_slots(response)
        assert slots == []

    def test_response_empty_choices(self):
        """A response with no choices yields no slots."""
        response = SimpleNamespace(choices=None)
        assert openai_adapter.response_slots(response) == []

    def test_parse_arguments_empty_and_invalid(self):
        assert openai_adapter.parse_arguments("") == {}
        assert openai_adapter.parse_arguments("{broken") == "{broken"
        assert openai_adapter.parse_arguments('{"a":1}') == {"a": 1}


class TestResponsesAdapterEdges:
    def test_request_input_content_string_item(self):
        """An input item whose content is a plain string."""
        data = {"input": [{"role": "user", "content": "hello"}]}
        slots = responses_adapter.request_input_slots(data)
        assert len(slots) == 1
        assert slots[0][0]() == "hello"

    def test_request_non_list_non_str_input_no_slots(self):
        data = {"input": 42}
        assert responses_adapter.request_input_slots(data) == []

    def test_response_output_items_string_content(self):
        """A response output item with string content (attribute)."""
        item = SimpleNamespace(content="out-text")
        response = SimpleNamespace(output=[item], output_text="")
        slots = responses_adapter.response_slots(response)
        assert any(s[0]() == "out-text" for s in slots)

    def test_response_output_text_only(self):
        response = SimpleNamespace(output_text="plain output", output=[])
        slots = responses_adapter.response_slots(response)
        assert len(slots) == 1
        assert slots[0][0]() == "plain output"
