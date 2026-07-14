# Changelog

This page mirrors [CHANGELOG.md](https://github.com/nitishagar/lexvault/blob/main/CHANGELOG.md).

## [Unreleased]

### Added
- Reversible proprietary-term pseudonymization engine (dictionary + regex
  detection, deterministic HMAC placeholders, leftmost-longest overlap
  resolution, local SQLite mapping vault).
- LiteLLM `CustomGuardrail` plugin with individual hooks only (no
  `apply_guardrail`) covering OpenAI + Anthropic-native, streaming
  (`ModelResponseStream` and raw Anthropic SSE bytes), and tool-call
  arguments.
- File-mount shim for LiteLLM's naive `split('.')` loader.
- 60-second Docker quickstart with a sample dictionary and config.
- MkDocs Material documentation site with auto-generated API reference.
- CI (lint + type-check + test matrix on Python 3.10–3.13 × litellm
  floor+latest) and Trusted-Publishing release workflow.
