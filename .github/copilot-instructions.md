# Project Guidelines

## Overview

ProtoFX converts ONNX models into PyTorch `torch.fx.GraphModule` objects. The goal is faithful, composable translation—not just inference parity but preserving graph structure so downstream `torch.compile`, quantization, and optimization passes work correctly.

## Code Style

- Python ≥ 3.12, type hints required on all public APIs
- Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with 120-char line limit
- Use `ruff` for linting/formatting (`ruff check` / `ruff format`)
- Prefer `match` statements over long `if/elif` chains for ONNX op dispatch
- All functions, classes, and methods must have [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- All documentation (`docs/`, docstrings, code comments) must be written in English

## Architecture

ProtoFX uses an ADR-centered documentation system.

- See [docs/adr/README.md](../docs/adr/README.md) for ADR process and index.
- See [docs/dev/ARCHITECTURE.md](../docs/dev/ARCHITECTURE.md) for architecture overview, document map, and authority order.
- See [docs/dev/IR.md](../docs/dev/IR.md) for the IR documentation hub.
- See [docs/dev/ir/](../docs/dev/ir/) documents for detailed IR specifications.
- See [docs/ROADMAP.md](../docs/ROADMAP.md) for planned milestones and feature status.
- See [docs/WORKBOARD.md](../docs/WORKBOARD.md) only when the user wants an execution checklist; it is not an architectural source of truth.

When documents disagree, use this precedence:

1. ADRs in `docs/adr/` for architecture decisions.
2. Specifications in `docs/dev/` for implementation-facing contracts.
3. `docs/ROADMAP.md` for milestone priority and scope.
4. `docs/WORKBOARD.md` for user-directed execution order only.

## Build and Test

```bash
# Install (editable)
pip install -e ".[dev]"

# Lint & format
ruff check src/ tests/
ruff format src/ tests/

# Test
pytest tests/ -v
```

## Conventions

- Each ONNX op handler is a function decorated with `@register_op("opname")` returning `torch.fx.Node`(s)
- Prefer raising `NotImplementedError` with the op name for unsupported ops rather than silent fallback
- Test each op handler with a minimal ONNX model fixture in `tests/ops/`
- Keep `torch` imports lazy inside emitter modules to speed up import time

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>
```

| Type | Usage |
|------|-------|
| `feat` | New feature (e.g., new op handler) |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or updating tests |
| `chore` | Build, CI, tooling changes |

- Scope examples: `ops`, `importer`, `emitter`, `ir`, `utils`
- Keep the description concise, lowercase, no period at the end
- Example: `feat(ops): add Conv op handler`
