# Integration

How to wire lexvault into a LiteLLM proxy deployment.

## Prerequisites

- [LiteLLM](https://github.com/BerriAI/litellm) proxy (`litellm --config ...`),
  version `>=1.80.15` (the iterator-hook dispatcher fix, PR #17626).
- Python `>=3.10`.
- lexvault installed (`pip install lexvault`) in the same environment as the proxy.

## Mount the shim

LiteLLM's guardrail loader does a naive `guardrail.split(".")` and loads
`<file>.py` from the config directory (`guardrail_registry.py`). A pip-installed
package can't be referenced directly, so you mount a one-line shim next to your
`config.yaml`:

```python
# lexvault_shim.py
from lexvault import LexVaultGuardrail as LexVaultGuardrail  # noqa: F401
```

Reference it as `guardrail: lexvault_shim.LexVaultGuardrail`. Every existing
third-party LiteLLM guardrail ships this shim.

## Minimal config

```yaml
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
      dictionary_path: /app/dictionary.yaml
      org_key: os.environ/LEXVAULT_ORG_KEY
```

## Docker

See the [examples directory](https://github.com/nitishagar/lexvault/tree/main/examples)
for a `docker-compose.yml` that volume-mounts `config.yaml`, `lexvault_shim.py`,
and `dictionary.yaml` into a lexvault-enabled LiteLLM image.

## Per-request config

Override config per request (without premium features) via `metadata` and/or
headers. lexvault reads **both** locations defensively because the metadata
variable name is route-dependent in LiteLLM:

| Route | Metadata location |
|---|---|
| `/v1/messages`, `/v1/responses`, batches, files | `data["litellm_metadata"]` |
| `/v1/chat/completions` (and everything else) | `data["metadata"]` |

### Scope override

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-lexvault-scope: team-alpha" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Project Titan"}],
    "metadata": {"requester_metadata": {"lexvault_scope": "team-alpha"}}
  }'
```

Supported keys: `lexvault_scope` (or header `x-lexvault-scope`).

!!! note
    `extra_body` / `get_guardrail_dynamic_request_body_params` is enterprise-premium-gated
    in LiteLLM and unusable for an OSS tool. lexvault uses the non-premium
    `metadata`/headers channel only.

## Guardrail ordering

LiteLLM runs guardrails in config-list order. If you run lexvault alongside a
**one-way MASK** guardrail (e.g. `litellm_content_filter` with `action: MASK`),
**list lexvault FIRST** so it masks before the one-way masker — otherwise the
one-way masker can destroy a term lexvault intended to reversibly map.

```yaml
guardrails:
  - guardrail_name: lexvault
    litellm_params:                                  # FIRST — reversible mask
      guardrail: lexvault_shim.LexVaultGuardrail
      mode: [pre_call, post_call]
      default_on: true
      # ...lexvault params...
  - guardrail_name: content-filter
    litellm_params:                                  # after — one-way filter
      guardrail: litellm_content_filter
      mode: [pre_call, post_call]
      default_on: true
      # ...content-filter params...
```

## Protected endpoints

Only the **unified routes** are protected:

- :material-check: `/v1/chat/completions`
- :material-check: `/v1/messages`
- :material-check: `/v1/responses`

Provider-prefixed **passthrough routes** bypass ALL guardrail hooks (they stream
via `PassThroughStreamingHandler.chunk_processor`):

- :material-close: `/anthropic/...`
- :material-close: `/openai/...`
- :material-close: `/azure/...`

Do not use passthrough routes if you need masking.

## Non-text call types

lexvault safely no-ops (skip + debug log) on non-text call types: `embeddings`,
`image_generation`, `transcription`, `rerank`, `speech`, `moderation`. With
`default_on: true` the guardrail runs on all call types, but non-text ones are
detected and skipped.

## Logging

lexvault's `async_logging_hook` re-masks the standard logging payload before
LiteLLM emits it to Langfuse/DataDog/OTel — originals never appear in logs, even
on the spend-logging path. This is defense-in-depth: the request is already
masked by `pre_call`, but the logging hook guards against any path that re-reads
raw kwargs/results.
