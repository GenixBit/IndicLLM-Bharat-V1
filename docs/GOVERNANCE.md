# Bharat AI Governance

## Contribution Model

- All development happens in the open via pull requests
- Every PR must pass CI (lint, type check, test)
- Every PR must be reviewed by at least one maintainer
- Maintainers are responsible for a specific area (tokenizer, training, eval, serving)

## Branch Strategy

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Stable, release-ready | Protected — CI required, review required |
| `develop` | Integration branch | CI required |
| `milestone-*` | Milestone-specific work | CI required |
| `feature/*` | Individual feature branches | None |
| `release/v*` | Release candidates | Review required |

## Decision Making

- Technical decisions: lazy consensus after 48h for comment on RFC-style PR
- Breaking changes: require explicit approval from area maintainer
- Data additions: require licence review
- Model releases: require safety review sign-off

## Communication

- GitHub Issues for bugs and feature requests
- GitHub Discussions for architecture and design decisions
- PR reviews for code changes

## Roles

| Role | Responsibility |
|------|---------------|
| Maintainer | Review PRs, approve releases, set technical direction |
| Contributor | Submit PRs, report issues, improve documentation |
| Reviewer | Provide technical review on PRs in their area |

## Code of Conduct

- Be respectful and inclusive
- Assume good faith
- Disagreements are technical, not personal
- Prioritise the project's health over individual preferences
