# Quickstart

Get lexvault masking your proprietary terms in under 2 minutes.

## 1. Install

```bash
pip install lexvault
```

lexvault needs Python 3.10+ and runs as a [LiteLLM](https://github.com/BerriAI/litellm)
proxy guardrail.

## 2. Create a dictionary

Define the proprietary terms you want masked. These are YOUR terms — codenames,
products, customers, schema names — not classic PII.

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

## 3. Configure LiteLLM

Mount the shim (LiteLLM's guardrail loader does a naive `split(".")` and loads
`<file>.py` from the config dir, so you need the one-line shim next to your config):

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
      guardrail: lexvault_shim.LexVaultGuardrail
      mode: [pre_call, post_call]
      default_on: true
      dictionary_path: dictionary.yaml
      org_key: os.environ/LEXVAULT_ORG_KEY
      scope: default
```

```python
# lexvault_shim.py (next to config.yaml)
from lexvault import LexVaultGuardrail as LexVaultGuardrail  # noqa: F401
```

## 4. Run

```bash
export OPENAI_API_KEY=sk-...
export LEXVAULT_ORG_KEY=$(openssl rand -hex 32)   # keep this secret + stable
litellm --config config.yaml
```

## 5. Verify

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "What is Project Titan?"}]
  }'
```

- The upstream LLM receives `[LEX-NZNH3BZX]`, **not** `Project Titan`.
- The client response has `Project Titan` **restored** — you never see the placeholder.

## Docker

The fastest way to try it without a real API key is the
[Docker quickstart](https://github.com/nitishagar/lexvault/tree/main/examples),
which boots LiteLLM + a fake backend that echoes requests so you can see the
mask/restore round-trip directly.

```bash
git clone https://github.com/nitishagar/lexvault.git
cd lexvault
docker compose -f examples/docker-compose.yml up --build
```

## Next steps

- [Concepts](concepts.md) — how masking, scopes, and deterministic placeholders work.
- [Configuration](config.md) — every config key.
- [Integration](integration.md) — LiteLLM proxy setup, per-request overrides, guardrail ordering.
