# Support

## Where to ask

- **Bugs and feature requests** → [GitHub Issues](https://github.com/nitishagar/lexvault/issues)
- **Questions and discussion** → [GitHub Discussions](https://github.com/nitishagar/lexvault/discussions)
- **Security vulnerabilities** → see [SECURITY.md](SECURITY.md) — **do not open a public issue**

## Before opening an issue

1. Search [existing issues](https://github.com/nitishagar/lexvault/issues) —
   yours may already be answered.
2. Check the [documentation](https://nitishagar.github.io/lexvault/),
   especially the [config reference](https://nitishagar.github.io/lexvault/config/)
   and [integration guide](https://nitishagar.github.io/lexvault/integration/).
3. Reproduce against the **latest released version** if you can.

## When opening a bug report

Include:

- lexvault version (`pip show lexvault`) and LiteLLM version.
- Your `config.yaml` (redact secrets / `org_key`) and a minimal dictionary.
- The call type and provider (e.g. `/v1/chat/completions` OpenAI,
  `/v1/messages` Anthropic-native, streaming vs non-streaming).
- What you expected vs. what happened.
- Logs (they should contain only placeholders, never originals — if you see an
  original term in a log, that's a bug, follow [SECURITY.md](SECURITY.md) to
  report it privately).
