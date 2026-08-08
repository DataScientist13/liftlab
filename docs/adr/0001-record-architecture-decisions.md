# ADR 0001 — Record architecture decisions

**Status:** accepted
**Date:** 2026-08-09

## Context

This repository makes a number of modeling choices that are not obvious and that a reader
should be able to interrogate: which adstock parameterisation, how priors are elicited, how a
geo-experiment posterior is turned into an MMM prior, what the optimizer is allowed to assume.
Without a record, those choices look arbitrary, and the reasoning is lost the moment it stops
being in anyone's head.

## Decision

Every non-obvious modeling or architectural decision gets a short ADR in `docs/adr/`, numbered
sequentially, using the standard Context / Decision / Consequences form. An ADR is written when the
decision is made, not retroactively before a release.

Superseded ADRs are not deleted or edited; a later ADR supersedes them and both stay in the
history.

## Consequences

- The README can stay short, because "why this way" lives here.
- Reviewers can disagree with a decision without reverse-engineering it from code.
- There is a small ongoing cost: a decision made in a hurry still needs its ADR before merge.
