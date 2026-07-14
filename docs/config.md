# Configuration

lexvault is configured via the `litellm_params` block of your LiteLLM
`config.yaml`. These keys are the public config contract — validated by a
pydantic model at construction, so a bad value fails fast with a clear error.

```yaml
guardrails:
  - guardrail_name: lexvault
    litellm_params:
      guardrail: lexvault_shim.LexVaultGuardrail
      mode: [pre_call, post_call]
      default_on: true
      dictionary_path: dictionary.yaml
      org_key: os.environ/LEXVAULT_ORG_KEY
      scope: default
      fail_open: false
      placeholder_format: "[LEX-{code}]"
      vault_path: ~/.lexvault/vault.db
      mask_system_role: true
      regex_terms: []
```

## Keys

### `dictionary_path`
{: data-toc-omit }

**Required.** Path to a YAML or JSON file with your proprietary terms.

```yaml
terms:
  - term: Project Titan
    type: codename
    case_sensitive: false      # optional, default false
regex_terms:
  - name: Employee ID
    pattern: 'EMP-\d{4,6}'
    type: id
    case_sensitive: false      # optional, default false
```

### `org_key`
{: data-toc-omit }

**Required.** Secret key used to derive deterministic placeholders. **Keep this
secret and stable.** Use `os.environ/LEXVAULT_ORG_KEY` to load from env. Changing
the key changes every placeholder for new masks (old mappings still restore under
the key they were created with, because restore looks up the vault by placeholder
string, not by re-deriving).

### `scope`
{: data-toc-omit }

Default: `default`. Isolates mappings. Two requests with different scopes get
independent placeholders for the same term. Override per-request via
`metadata.requester_metadata.lexvault_scope` (see [integration](integration.md#per-request-config)).

### `fail_open`
{: data-toc-omit }

Default: `false`. On a masking/restoring error:

- `false` (recommended) — **block**: mask error → HTTP 503; restore error →
  sanitized HTTP 200. No original reaches upstream, no placeholder reaches client.
- `true` — **pass through**: the request/response goes through unmasked. Use only
  if you accept the leak risk in exchange for availability.

### `placeholder_format`
{: data-toc-omit }

Default: `[LEX-{code}]`. The format must contain `{code}`. The `{code}` is 8
base32 chars derived from the HMAC. A collision (two distinct terms hashing to
the same truncated code) appends a `-2`, `-3` … suffix.

### `vault_path`
{: data-toc-omit }

Default: `~/.lexvault/vault.db`. Path to the local SQLite vault. The parent dir
is created `0700` and the file `0600` (it contains the original terms). No
network code; the vault never leaves the host.

### `mask_system_role`
{: data-toc-omit }

Default: `true`. Whether to mask the `system` role / Anthropic `system` prompt.
Set `false` if your system prompt contains dictionary terms that must reach the
model verbatim for instruction (documented risk).

### `regex_terms`
{: data-toc-omit }

Default: `[]`. Additional regex terms (merged with any `regex_terms` in the
dictionary file). Each entry: `{name, pattern, type, case_sensitive}`.

### Base-class mask flags (`mask_request_content` / `mask_response_content`)
{: data-toc-omit }

The LiteLLM `CustomGuardrail` base class exposes `mask_request_content` and
`mask_response_content` boolean knobs. lexvault **ignores these** — it always
masks/restores based on its own hook implementations (invariant 4: lexvault
does not define `apply_guardrail`). Setting either flag has no effect.

## Per-request overrides

See [integration → per-request config](integration.md#per-request-config) for
overriding `scope` per request via `metadata` or headers.

## Environment-variable placeholders

LiteLLM convention: any string value of the form `os.environ/NAME` is resolved
from the environment at guardrail construction. Use this for `org_key` and any
path you want to keep out of the config file.
