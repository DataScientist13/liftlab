# ADR 0003 — The calibration bridge is a likelihood term, not a prior on beta

**Status:** accepted
**Date:** 2026-08-09

## Context

[ADR 0002](0002-mmm-parameterisation.md) recorded the measured problem: the MMM recovers carryover
well but biases channel effect sizes upward by roughly a quarter, because effect size is identified
by the *level* of the response, which trades off against the organic baseline. Aggregate revenue
does not contain the information needed to separate them.

An incrementality experiment does contain it. The question is how to get that information into the
model.

The obvious reading of "geo-lift posteriors become informative priors on MMM channel coefficients"
is to derive a prior on `beta_c` directly from the experiment: convert measured lift into an
implied coefficient, and centre a prior there.

## Decision

The experiment enters as an **additional likelihood term**, not as a prior on `beta_c`:

```
predicted_c = sum over the test window of  beta_c * hill(adstock(spend_ct))  * revenue_scale
measured_c ~ Normal(predicted_c, standard_error_c)
```

`liftlab` does not run or analyse the experiment. It consumes a result — incremental revenue with a
standard error — from wherever the test was measured.

## Why not a prior on beta

Converting a lift measurement into a statement about `beta_c` requires knowing how much saturated,
adstocked response the channel generated during the test window. That runs through `decay`,
`half_saturation`, and `slope` — parameters that are themselves uncertain, and whose posteriors are
correlated with `beta_c`.

Deriving a prior on `beta_c` therefore means fixing those nuisance parameters at point estimates and
propagating no uncertainty from them. The resulting prior would be too narrow, and wrong in a
direction that depends on how badly the point estimates missed.

Expressing the experiment as a likelihood on the quantity it actually measured — incremental revenue
over a window — lets NUTS propagate its information through the whole mapping. Every parameter the
mapping touches gets updated jointly. The experiment constrains a function of the parameters, which
is exactly what it observed, rather than a parameter it never saw directly.

## Consequences

- The standard error does the work. A noisy experiment barely moves the posterior; a precise one
  moves it a lot. This is tested directly: the same claimed effect with a 5% standard error pulls
  the estimate materially further than with a 200% one.
- Multiple experiments on the same channel compose without special handling — each is another
  observation.
- The bridge inherits the experiment's validity. It treats the estimate as unbiased and its
  standard error as honest, and will propagate a biased geo test into the MMM faithfully. That
  limitation is stated in the README rather than mitigated, because there is no way to detect it
  from the aggregate data.
- Because the term is a likelihood, it can be checked the same way any other fit is: the model
  exposes `experiment_predicted` as a deterministic site, so the implied incremental revenue can be
  compared against what the experiment reported.
