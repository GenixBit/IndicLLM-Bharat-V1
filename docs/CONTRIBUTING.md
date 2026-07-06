# Contributing to Bharat AI

## Getting Started

```bash
git clone https://github.com/GenixBit/IndicLLM-Bharat-V1.git
cd IndicLLM-Bharat-V1
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

1. Pick an issue from the milestone or open a new one
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests first (red-green-refactor)
4. Implement the feature
5. Run all checks: `ruff check .` + `mypy bharat/` + `pytest tests/`
6. Commit with a descriptive message
7. Push and open a PR

## Running Checks

```bash
# Lint
ruff check .

# Type check
mypy bharat/

# Tests
pytest tests/ -v

# All checks at once
pre-commit run --all-files
```

## Code Style

- Python 3.11+ with `from __future__ import annotations`
- Type hints on all public functions
- 100 character line limit
- Ruff for linting and formatting
- Descriptive variable names (not single-letter except in math-heavy contexts)
- Legacy code stays in place until migration is verified — never delete working code

## Pull Request Guidelines

1. PR title follows conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `refactor:`
2. PR description explains the change and links to any related issues
3. All CI checks pass before merge
4. At least one maintainer review required
5. No force pushes to shared branches

## Test Guidelines

- Every bug fix includes a regression test
- Every new feature includes tests for success and error cases
- Test file mirrors source path: `bharat/tokenizer/loader.py` → `tests/test_tokenizer.py`
- Use `pytest` fixtures for shared resources
- Mark GPU-dependent tests with `@pytest.mark.gpu`
- Mark slow tests with `@pytest.mark.slow`

## Documentation

- All public modules have docstrings
- README only contains verified, working instructions
- Architecture decisions documented in `docs/`
- Model cards accompany every release

## Getting Help

- Open a GitHub Issue for bugs
- Start a GitHub Discussion for design questions
- Tag maintainers for review on PRs
