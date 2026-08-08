---
description: Run the parameter-recovery benchmark and summarize bias and posterior coverage.
allowed-tools: Bash(just recover), Bash(uv run:*), Read, Glob
---

Run the recovery benchmark and report the result.

1. `just recover`
2. Read the emitted report under `reports/recovery/`.
3. Summarize, per scenario: parameter bias (relative, signed), posterior coverage of the nominal
   interval, and sampler diagnostics.
4. Flag anything that fails the gate: coverage outside the stated tolerance of nominal, or bias
   beyond the documented threshold.
5. If the README's coverage table is now stale, say so and show the diff — do not silently update
   published numbers.

Report what actually ran. If the benchmark did not complete, say that rather than summarizing a
partial run as if it were finished.
