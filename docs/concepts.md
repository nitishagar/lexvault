# Concepts

## The trust boundary

lexvault enforces a hard boundary: **your proprietary terms never leave your
enterprise boundary on the way to the LLM, and placeholders never reach your
client on the way back.**

```
client ──"Project Titan"──▶ [lexvault masks] ──▶"[LEX-NZNH3BZX]"──▶ upstream LLM
client ◀─"Project Titan"── [lexvault restores] ◀─"[LEX-NZNH3BZX]"◀── upstream LLM
```

The mapping vault that ties placeholders to originals lives locally (SQLite, by
default at `~/.lexvault/vault.db`, mode `0600`). No network egress.

## Mask and restore

- **Mask** runs on the request path (`async_pre_call_hook`). It scans the text
  for dictionary + regex terms using an Aho-Corasick automaton (built once at
  construction, sub-millisecond per request), replaces each match with a
  deterministic placeholder, and records the mapping in the vault.
- **Restore** runs on the response path (`async_post_call_success_hook` for
  non-streaming, `async_post_call_streaming_iterator_hook` for streaming). It
  finds placeholder-namespace spans and replaces them with the originals from
  the vault.

Both run on all surfaces: message content, tool-call arguments, tool results,
and across OpenAI and Anthropic-native shapes.

## Deterministic placeholders

A placeholder is derived from the term + your secret `org_key` + a `scope` via
HMAC-SHA256 → base32. Given the same inputs, the placeholder is **always
identical**:

```
Project Titan + org_key + scope="default" → [LEX-NZNH3BZX]
```

This gives you:

- **Cross-turn consistency** — the same codename maps to the same placeholder
  across every turn in a conversation, so the model reasons coherently.
- **Cross-surface consistency** — a term in the user message, a `tool_calls`
  argument, and a tool result all share one placeholder.
- **Reversibility without the original** — restore looks up the vault by
  placeholder string; it doesn't need the original term.

Keep the `org_key` secret. If it leaks, placeholders can be brute-forced back to
originals. Treat key rotation as an operational concern (see [config](config.md#org_key)).

## Scopes

A `scope` isolates mappings. Two teams using the same lexvault instance but
different scopes get independent placeholders for the same term (useful when the
same word is a codename in one team and a common word in another). The default
scope is `default`.

## Fail-closed by default

If masking or restoring fails (vault unreadable, unexpected response shape),
lexvault **blocks** rather than leaks:

- A **mask error** blocks the request (HTTP 503) — no original reaches the
  upstream LLM.
- A **restore error** returns a sanitized response (HTTP 200,
  `finish_reason="content_filter"`) containing **no placeholder and no original**.

Set `fail_open: true` to pass through instead (documented risk; not recommended
for production).

## Idempotent re-masking

If a request contains an already-masked placeholder (e.g. conversation history
from a previous turn), lexvault leaves it untouched — it pre-excludes
placeholder-namespace spans from detection. No double-masking, no nesting.

## What lexvault is NOT

- **Not a PII detector.** It masks your dictionary terms, not names/emails/SSNs
  auto-discovered via NER. (NER/embedding detection is a v0.2 roadmap item.)
- **Not irreversible.** The mapping is reversible by design (it's the product's
  value). If the `org_key` leaks, placeholders can be reversed — this is a
  documented risk, not a bug.
- **Not a standalone proxy.** v0.1 is a LiteLLM guardrail plugin. A standalone
  OpenAI-compatible ASGI mode is on the v0.2 roadmap.

## Known boundaries

- **Provider-prefixed passthrough routes** (`/anthropic/...`, `/openai/...`)
  bypass all guardrail hooks in LiteLLM. Only the unified routes
  (`/v1/chat/completions`, `/v1/messages`, `/v1/responses`) are protected. See
  [integration](integration.md#protected-endpoints).
- **Tool definitions** in `data["tools"]` are not masked — the model needs
  verbatim schemas for correct function dispatch.
- **Vault growth** — mappings persist (no TTL in v0.1), so the vault grows over
  time bounded by distinct terms × requests. A future cleanup/TTL pass is on the
  roadmap.
