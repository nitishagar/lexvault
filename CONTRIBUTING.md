# Contributing to lexvault

Thanks for your interest in improving lexvault! This is a small, focused
project and we welcome contributions of all sizes.

## Quick links

- [Issues](https://github.com/nitishagar/lexvault/issues) — bugs and feature ideas.
- [Discussions](https://github.com/nitishagar/lexvault/discussions) — questions and design talk.
- [Security reports go here, not in public issues](SECURITY.md).

## Development setup

lexvault requires **Python 3.10+** and uses [`uv`](https://docs.astral.sh/uv/)
for environment management (any standard venv + pip works too).

```bash
git clone https://github.com/nitishagar/lexvault.git
cd lexvault

# Create a 3.10+ venv (uv is fastest)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install the package in editable mode with dev tools
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
uv pip install ruff mypy pytest pytest-asyncio
```

## Day-to-day commands

```bash
# Lint + format
ruff check src tests
ruff format src tests

# Type-check the package (strict)
mypy src/lexvault

# Run the test suite
pytest -v
```

Pre-commit hooks (optional but recommended):

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Code style

- **Type hints everywhere** in `src/lexvault` — `mypy --strict` must pass.
- **`ruff`** handles lint + format; the config lives in `pyproject.toml`.
- Match the surrounding code's style. Inconsistency is a compounding cost.
- Public functions and classes get docstrings.

## Tests

- **Unit tests** (no LiteLLM dependency) live in `tests/unit/` and cover the
  masking engine, detector, vault, and streaming buffer invariants.
- **Integration tests** live in `tests/integration/` and run against a fake
  OpenAI backend through a real LiteLLM proxy.
- Every public behavior and every edge case (overlaps, collisions,
  concurrency, fail-closed, boundary splits) should have a test. Tests that
  pass but assert nothing are worse than none — assert on real outcomes.

## Commit and pull request conventions

- **Commit messages:** write a clear summary line (`<area>: <imperative>`),
  then a body explaining *why*. Reference the issue number if applicable.
- **DCO / sign-off:** by submitting a pull request you certify that your
  contributions are your own and licensed under the Apache-2.0 license. Please
  add the `Signed-off-by: Your Name <email>` line to your commits
  (`git commit -s`) to record this.
- Keep pull requests focused; one logical change per PR makes review faster.
- Don't widen scope beyond your PR's goal — note adjacent ideas in an issue.

## Releasing

Releases are cut from `main` by tagging `vX.Y.Z` (SemVer). The release
workflow builds the wheel + sdist and publishes to PyPI via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC). See
[CHANGELOG.md](CHANGELOG.md) for the release history.

## Code of conduct

Participation in this project is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please be
excellent to each other.
