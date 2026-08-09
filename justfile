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
# Pass --profile full for the publication run, --check to fail on coverage drift.
recover *ARGS:
    uv run python -m liftlab.recovery {{ARGS}}

# Pair the uncalibrated and calibrated benchmark arms over their common healthy seeds.
compare:
    uv run python -m liftlab.recovery --profile calibrated \
        --compare docs/recovery/full-raw.csv docs/recovery/calibrated-raw.csv \
        --output docs/recovery

all: lint test
