default:
    @just --list

# Cold-clone bootstrap: deps, pre-commit, and the attribution git hook.
setup:
    uv sync --all-extras --dev
    uv run pre-commit install
    ./scripts/install-git-hooks.sh

test:
    uv run pytest

lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy src

fix:
    uv run ruff check --fix .
    uv run ruff format .

# Parameter-recovery benchmark: simulate a known DGP, fit, report bias and coverage.
recover:
    uv run python -m liftlab.recovery

all: lint test
