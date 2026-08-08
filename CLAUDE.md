# liftlab

Bayesian marketing mix modeling calibrated by geo-incrementality experiments, with a budget
optimizer that carries posterior uncertainty into the recommendation.

## Domain context

Post-ATT attribution is broken. Last-click ROAS systematically misattributes incremental revenue,
and MMM alone is only weakly identified — many parameter combinations fit the same aggregate
series. The fix is to calibrate MMM against randomized or quasi-randomized geo experiments, which
is what Meta and Google do internally.

The pieces:

- **MMM core** — geometric and Weibull adstock, Hill saturation, Fourier seasonality,
  hierarchical priors across channels, informative ROAS / budget-share priors.
- **Geo-lift** — matched-market selection, synthetic control / Bayesian structural time series,
  power analysis and MDE calculators for designing a test *before* it runs.
- **Calibration bridge** — geo-lift posteriors become informative priors on MMM channel
  coefficients. This is the piece that separates practitioners from whitepaper readers.
- **Budget optimizer** — constrained allocation over fitted response curves, propagating posterior
  uncertainty rather than optimizing point estimates.
- **Recovery harness** — simulate a known DGP, fit, report parameter bias and posterior coverage
  across a scenario grid. The coverage table goes in the README.

**Demo data is synthetic Israeli e-commerce**: Hebrew channel names, Israeli holiday calendar
(Pesach and Rosh Hashana retail spikes, Yom Kippur hard zeroes, Ramadan in mixed markets),
Saturday-trough weekly seasonality. The DGP is documented, so ground truth is knowable — that is
what makes the recovery benchmark meaningful.

## Conventions

- Python 3.12 via `uv`. Never `pip install` into the system interpreter.
- `src/` layout. Public API surface only through `src/liftlab/__init__.py`.
- `ruff` + `mypy --strict` must pass before any commit.
- Conventional Commits.
- Every non-obvious modeling decision gets an ADR in `docs/adr/`.
- Priors are documented where they are declared, with the reasoning for the scale.

## Commands

- `just setup`   — cold-clone bootstrap
- `just test`    — full suite
- `just lint`    — ruff + mypy
- `just fix`     — autofix lint and format
- `just recover` — parameter-recovery benchmark (the repo-specific CI gate)

## Do NOT

- **Never** add `Co-Authored-By: Claude`, `Generated with Claude Code`, or 🤖 to a commit
  message. Enforced by a `PreToolUse` hook and a git `commit-msg` hook; do not work around either.
- Never commit data files >2 MB. Regenerate synthetic data from the seeded generator instead.
- Never reference, read from, or copy client work into this repo. All data here is synthetic or
  public.
- Never weaken a test to make CI pass. Fix the code or state the limitation.
- Never report a metric in the README that isn't reproducible by a named command.
- Never report a recovery or coverage number that has not actually been run.
