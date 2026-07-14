# lexvault

> Reversible proprietary-term pseudonymization for LiteLLM. Mask your codenames,
> products, customers, and schema on the way out — restore them faithfully on
> the way back.

[![CI](https://github.com/nitishagar/lexvault/actions/workflows/ci.yml/badge.svg)](https://github.com/nitishagar/lexvault/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/lexvault.svg)](https://pypi.org/project/lexvault/)
[![Python versions](https://img.shields.io/pypi/pyversions/lexvault.svg)](https://pypi.org/project/lexvault/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://nitishagar.github.io/lexvault/)

**lexvault** is a [LiteLLM](https://github.com/BerriAI/litellm) proxy plugin that
swaps an enterprise's *own* dictionary terms (codenames, unreleased products,
customer names, internal identifiers — **not** classic PII) for deterministic
placeholders on the request path and restores them on the response path. It
works across:

- ✅ Non-streaming OpenAI `ModelResponse` (`/v1/chat/completions`)
- ✅ Non-streaming Anthropic-native content blocks + `tool_use.input`
  (`/v1/messages`)
- ✅ OpenAI `tool_calls[].function.arguments` (request history + output)
- ✅ Streaming OpenAI `ModelResponseStream`
- ✅ Streaming Anthropic-native **raw SSE bytes** (re-framed + restored)
- ✅ `/v1/responses` text (Responses API)

with **zero un-restored placeholders reaching the client** and **zero originals
reaching the upstream LLM or logs**.

## Why?

Classic PII redaction is saturated. The wedge lexvault fills is *reversible*
pseudonymization for an enterprise's **proprietary** knowledge, with correct
round-trip handling across tool calls and streaming — exactly where the
leading gateway's own path fails ([BerriAI/litellm#22821](https://github.com/BerriAI/litellm/issues/22821)).

The trust boundary is explicit: your dictionary and the mapping vault never
leave your boundary by default. See the [concepts docs](https://nitishagar.github.io/lexvault/concepts/)
for the design and threat model.

## 60-second quickstart

```bash
pip install lexvault
```

Mount the shim next to your LiteLLM `config.yaml` (LiteLLM's guardrail loader
does a naive `split('.')` and loads `<file>.py` from the config dir, so you need
the one-line shim):

```yaml
# config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: lexvault
    litellm_params:
      guardrail: lexvault_shim.LexVaultGuardrail   # references the mounted shim
      mode: [pre_call, post_call]
      default_on: true
      dictionary_path: dictionary.yaml
      org_key: os.environ/LEXVAULT_ORG_KEY
      scope: default
```

```yaml
# dictionary.yaml
terms:
  - term: Project Titan
    type: codename
  - term: customer_database
    type: schema
regex_terms:
  - name: Employee ID
    pattern: 'EMP-\d{4,6}'
    type: id
```

Run the proxy, and any request mentioning `Project Titan` reaches the upstream
LLM as something like `[LEX-AB12CD34]` — and the response is restored before it
reaches your client.

See [`examples/`](examples/) for a complete Docker quickstart.

## Distribution shape

```
pip install lexvault        # the engine + guardrail
                            # + lexvault_shim.py mounted next to config.yaml
```

## Documentation

Full docs (concepts, config reference, API reference, integration guide) live
at **[nitishagar.github.io/lexvault](https://nitishagar.github.io/lexvault/)**.

## What's NOT in v0.1

- NER / embeddings / GLiNER / spaCy detection (v0.2)
- Standalone OpenAI-compatible ASGI proxy mode (v0.2)
- Redis vault backend / multi-instance coordination (v0.2)
- OTel/JSONL exhaust recorder, `scan` CLI, TS SDK, UI, auth/mTLS (v0.2)
- At-rest vault encryption (file-mode `0600` protection + documented risk for v0.1)
- Tool *definition* masking in `data["tools"]` (the model needs verbatim schemas)
- Provider-prefixed passthrough routes (`/anthropic/...`) bypass all guardrail
  hooks — documented as unprotected; use unified endpoints (`/v1/*`)

## Prior art

`palena-litellm-pseudonymizer` and similar one-off pseudonymizers exist, but
none package a focused, tool-call + streaming-correct, enterprise-dictionary
reversible round-trip as a drop-in LiteLLM plugin. LiteLLM's built-in
`litellm_content_filter` `MASK` is one-way (no restore), and its Presidio path
has the open round-trip bug above. See the docs for the full landscape.

## License

[Apache-2.0](LICENSE). Bundles/depends on [pyahocorasick](https://github.com/WojciechMula/pyahocorasick)
(BSD-3-Clause), [litellm](https://github.com/BerriAI/litellm) (MIT), and
[pydantic](https://github.com/pydantic/pydantic) (MIT). See [NOTICE](NOTICE).
