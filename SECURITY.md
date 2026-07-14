# Security Policy

lexvault is a security-adjacent tool: its entire purpose is to keep
proprietary terms out of upstream LLM providers and logs. We take correctness
and disclosure handling seriously.

## Supported Versions

Security fixes are applied to the latest released version on PyPI and the
`main` branch. Backports to older versions are provided on a best-effort basis.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for a security vulnerability.**

Please report vulnerabilities privately using GitHub's built-in vulnerability
reporting:

1. Go to **[Report a vulnerability](https://github.com/nitishagar/lexvault/security/advisories/new)**.
2. Provide a description of the issue, steps to reproduce, and the impact.

You may also email reports to: **nitishagar@users.noreply.github.com** with the
subject line `[lexvault security]`.

### What to include

- A clear description of the vulnerability and its security impact.
- The lexvault version and LiteLLM version you tested against.
- A minimal reproducer (config.yaml + dictionary + request, with placeholder
  secrets — **never send real proprietary terms or secrets**).

### Response timeline

We aim to acknowledge reports within **72 hours** and to provide an initial
assessment within **7 days**. Coordinated disclosure and credit are the
default; we follow responsible-disclosure practice and will publish a GitHub
Security Advisory with a CVE where applicable once a fix is available.

## Threat model notes

lexvault is a **fail-closed** guardrail: by default (`fail_open: false`) a
masking or restore error blocks the request/response rather than allowing an
unmasked term to leak. Please report any path where a masked term could reach
the upstream provider or the client.

The local mapping vault (default `~/.lexvault/vault.db`) contains the original
(unmasked) terms and is created with file mode `0600`. Treat it as sensitive.
At-rest encryption of the vault is on the roadmap (see the docs); until then,
rely on filesystem permissions and disk encryption.

The `org_key` used to derive placeholders should be kept secret. If it leaks,
deterministic placeholders can be brute-forced back to their originals — treat
key rotation as a documented operational concern.
