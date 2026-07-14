# Changelog

All notable changes to lexvault are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[Unreleased]: https://github.com/nitishagar/lexvault/compare/main...HEAD
