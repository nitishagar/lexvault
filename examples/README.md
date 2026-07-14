# lexvault examples

A 60-second quickstart with LiteLLM + a fake backend in Docker.

## What's here

- [`config.yaml`](config.yaml) — a LiteLLM proxy config with the lexvault guardrail mounted.
- [`lexvault_shim.py`](lexvault_shim.py) — the file-mount shim LiteLLM's loader needs.
- [`dictionary.yaml`](dictionary.yaml) — sample proprietary terms (`Project Titan`, `customer_database`, an `EMP-\d{4,6}` regex).
- [`fake_backend.py`](fake_backend.py) — a tiny stdlib echo server that mimics an OpenAI chat completion (no deps, no network).
- [`Dockerfile.fake-backend`](Dockerfile.fake-backend) — builds the fake backend into a small image.
- [`docker-compose.yml`](docker-compose.yml) — boots LiteLLM + the fake backend.
- [`Dockerfile.litellm`](Dockerfile.litellm) — builds a lexvault-enabled LiteLLM image (installs lexvault from the repo source).

## 60-second quickstart

```bash
# 1. Boot LiteLLM + the fake backend.
docker compose -f examples/docker-compose.yml up --build

# 2. In another terminal, send a request mentioning a codename.
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fake",
    "messages": [{"role": "user", "content": "What is Project Titan and where is EMP-123456?"}]
  }'
```

**What you'll see:**
- The fake backend (which echoes its input) receives `[LEX-...]` placeholders, **not** `Project Titan` or `EMP-123456` — the originals never leave your boundary.
- The client response has the originals **restored** — you see `Project Titan` and `EMP-123456`, never the placeholders.

## Configuring for a real LLM

Edit [`config.yaml`](config.yaml) and replace the `fake` model with a real one:

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
```

Then set your API key and the lexvault org key:

```bash
export OPENAI_API_KEY=sk-...
export LEXVAULT_ORG_KEY=$(openssl rand -hex 32)
docker compose -f examples/docker-compose.yml up --build
```

## Adding your own terms

Edit [`dictionary.yaml`](dictionary.yaml):

```yaml
terms:
  - term: Project Titan          # a literal codename
    type: codename
  - term: customer_database      # a schema/table name
    type: schema
regex_terms:
  - name: Employee ID            # a structured internal ID
    pattern: 'EMP-\d{4,6}'
    type: id
```

See the [config reference](https://nitishagar.github.io/lexvault/config/) for every option.

## Notes

- **Protected endpoints:** lexvault only protects the unified routes (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`). Provider-prefixed passthrough routes (`/anthropic/...`, `/openai/...`) bypass all guardrail hooks — don't use them if you need masking. See the [integration guide](https://nitishagar.github.io/lexvault/integration/).
- **The org key** derives deterministic placeholders. Keep it secret and stable — changing it changes every placeholder, so old mappings restore under the key they were created with (see [concepts](https://nitishagar.github.io/lexvault/concepts/)).
- **The vault** (`~/.lexvault/vault.db` by default, or the container path you configure) stores the original terms. It's created `0600` — treat it as sensitive.
