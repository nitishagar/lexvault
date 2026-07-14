---
template: home.html
title: lexvault
hide:
  - navigation
  - toc
---

# lexvault

> Reversible proprietary-term pseudonymization for LiteLLM. Mask your codenames,
> products, customers, and schema on the way out — restore them faithfully on
> the way back.

lexvault is a [LiteLLM](https://github.com/BerriAI/litellm) proxy plugin that
swaps an enterprise's *own* dictionary terms for deterministic placeholders on
the request path and restores them on the response path — with **zero un-restored
placeholders reaching the client** and **zero originals reaching the upstream LLM
or logs**.

## The trust boundary

Classic PII redaction is saturated. lexvault fills the wedge that incumbents
miss: *reversible* pseudonymization for an enterprise's **proprietary** knowledge,
with correct round-trip handling across **tool calls** and **streaming** — exactly
where the leading gateway's own path fails
([BerriAI/litellm#22821](https://github.com/BerriAI/litellm/issues/22821)).

Your dictionary and mapping vault never leave your boundary by default.

## 60-second quickstart

```bash
pip install lexvault
```

```yaml
# config.yaml — mount next to lexvault_shim.py
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: lexvault
    litellm_params:
      guardrail: lexvault_shim.LexVaultGuardrail
      mode: [pre_call, post_call]
      default_on: true
      dictionary_path: dictionary.yaml
      org_key: os.environ/LEXVAULT_ORG_KEY
```

```bash
litellm --config config.yaml
```

→ A request mentioning `Project Titan` reaches the upstream LLM as
`[LEX-NZNH3BZX]`, and the response is restored before it reaches your client.
[See the full quickstart →](quickstart.md)

## Feature surface

- :material-check-circle: Non-streaming OpenAI (`/v1/chat/completions`)
- :material-check-circle: Non-streaming Anthropic-native content blocks + `tool_use.input` (`/v1/messages`)
- :material-check-circle: OpenAI `tool_calls[].function.arguments`
- :material-check-circle: Streaming OpenAI `ModelResponseStream`
- :material-check-circle: Streaming Anthropic-native **raw SSE bytes** (re-framed + restored)
- :material-check-circle: `/v1/responses` text (Responses API)
- :material-close-circle: NER / embeddings / GLiNER detection (v0.2)
- :material-close-circle: Standalone ASGI proxy mode (v0.2)

[Explore the concepts →](concepts.md){ .md-button .md-button--primary }
[Docker quickstart →](https://github.com/nitishagar/lexvault/tree/main/examples){ .md-button }
