"""Tests for the recovery benchmark harness.

The benchmark is the repository's headline claim, so the harness itself needs testing: a
scoring bug would produce a table that looks authoritative and is wrong. These tests exercise
the scoring, aggregation, and gate logic on synthetic inputs, and run the sampler only once at
the smallest useful size.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from liftlab.dgp import DEFAULT_CHANNELS, DGPConfig, generate_panel
from liftlab.model import prepare_data
from liftlab.recovery import (
    BENCHMARK_CHANNELS,
    CALIBRATED,
    FAST,
    FULL,
    PROFILES,
    BenchmarkProfile,
    _markdown_table,
    _simulate_experiments,
    _truth_table,
    check_gate,
    format_comparison,
    format_report,
    main,
    paired_comparison,
    summarise,
)

TINY = BenchmarkProfile(
    name="tiny",
    seeds=(0,),
    n_periods=120,
    num_warmup=30,
    num_samples=30,
    num_chains=2,
    coverage_tolerance=1.0,
    max_r_hat=99.0,
    min_ess_bulk=0.0,
    # A 30-draw fit diverges freely; this profile exists to exercise the plumbing.
    max_divergence_rate=1.0,
)


def _fake_results(coverage_95: float, *, n: int = 20, parameter: str = "roas") -> pd.DataFrame:
    """Build a results frame with a known coverage rate, for testing aggregation."""
    covered = [i < round(coverage_95 * n) for i in range(n)]
    return pd.DataFrame(
        {
            "seed": range(n),
            "channel": ["search"] * n,
            "parameter": [parameter] * n,
            "truth": [2.0] * n,
            "posterior_mean": [2.2] * n,
            "posterior_median": [2.1] * n,
            "relative_bias": [0.1] * n,
            "relative_bias_median": [0.05] * n,
            "covered_95": covered,
            "covered_50": covered,
            "interval_width_95": [1.0] * n,
            "max_r_hat": [1.0] * n,
            "min_ess_bulk": [500.0] * n,
            "divergences": [0] * n,
            "divergence_rate": [0.0] * n,
            "seconds": [1.0] * n,
        }
    )


class TestProfiles:
    def test_registry_exposes_every_profile(self):
        assert PROFILES == {"fast": FAST, "full": FULL, "calibrated": CALIBRATED}

    def test_calibrated_arm_matches_the_baseline_arm_apart_from_experiments(self):
        """The two published arms must differ only by the experiment likelihood."""
        assert CALIBRATED.seeds == FULL.seeds
        assert CALIBRATED.n_periods == FULL.n_periods
        assert CALIBRATED.num_warmup == FULL.num_warmup
        assert CALIBRATED.num_samples == FULL.num_samples
        assert CALIBRATED.num_chains == FULL.num_chains
        assert CALIBRATED.target_accept_prob == FULL.target_accept_prob
        assert FULL.calibrated_channels == ()
        assert CALIBRATED.calibrated_channels == ("search", "social")

    def test_fast_profile_disables_the_coverage_gate(self):
        # Two seeds cannot estimate coverage; asserting on it would be theatre.
        assert FAST.coverage_tolerance >= 1.0
        assert len(FAST.seeds) < 5

    def test_full_profile_has_enough_seeds_to_estimate_coverage(self):
        assert len(FULL.seeds) >= 10
        assert FULL.n_periods >= 1_000
        assert FULL.num_chains >= 2

    def test_single_chain_profile_is_rejected(self):
        """One chain makes R-hat NaN, which silently excludes every replication."""
        with pytest.raises(ValueError, match="num_chains"):
            BenchmarkProfile(
                name="broken", seeds=(0,), n_periods=10, num_warmup=1, num_samples=1, num_chains=1
            )

    def test_seedless_profile_is_rejected(self):
        with pytest.raises(ValueError, match="at least one seed"):
            BenchmarkProfile(name="broken", seeds=(), n_periods=10, num_warmup=1, num_samples=1)

    def test_benchmark_channels_cover_every_dgp_channel(self):
        assert [c.slug for c in BENCHMARK_CHANNELS] == [c.slug for c in DEFAULT_CHANNELS]
        assert sum(c.adstock == "weibull" for c in BENCHMARK_CHANNELS) == 1


class TestTruthTable:
    @pytest.fixture
    def data(self):
        panel = generate_panel(DGPConfig(n_periods=120, seed=1))
        return panel, prepare_data(panel.data, BENCHMARK_CHANNELS)

    def test_scales_half_saturation_to_model_units(self, data):
        panel, prepared = data
        rows = _truth_table(panel.truth.roas, prepared)
        row = next(
            r for r in rows if r["channel"] == "search" and r["parameter"] == "half_saturation"
        )
        expected = DEFAULT_CHANNELS[0].half_saturation / prepared.spend_scale[0]
        assert row["truth"] == pytest.approx(expected)

    def test_scales_beta_by_mean_revenue(self, data):
        panel, prepared = data
        rows = _truth_table(panel.truth.roas, prepared)
        row = next(r for r in rows if r["channel"] == "email" and r["parameter"] == "beta")
        expected = DEFAULT_CHANNELS[4].coefficient / prepared.revenue_scale
        assert row["truth"] == pytest.approx(expected)

    def test_roas_truth_is_passed_through_in_currency_units(self, data):
        panel, prepared = data
        rows = _truth_table(panel.truth.roas, prepared)
        row = next(r for r in rows if r["channel"] == "tv" and r["parameter"] == "roas")
        assert row["truth"] == pytest.approx(panel.truth.roas["tv"])

    def test_decay_positions_index_only_geometric_channels(self, data):
        panel, prepared = data
        rows = [r for r in _truth_table(panel.truth.roas, prepared) if r["parameter"] == "decay"]
        # TV is Weibull, so four decays with contiguous positions 0..3.
        assert len(rows) == 4
        assert [r["position"] for r in rows] == [0, 1, 2, 3]
        assert "tv" not in [r["channel"] for r in rows]

    def test_every_channel_gets_the_shared_parameters(self, data):
        panel, prepared = data
        rows = _truth_table(panel.truth.roas, prepared)
        for name in ("roas", "beta", "slope", "half_saturation"):
            assert sum(r["parameter"] == name for r in rows) == len(DEFAULT_CHANNELS)


class TestSummarise:
    def test_coverage_is_computed_as_a_fraction_of_replications(self):
        summary = summarise(_fake_results(0.8, n=20), FULL)
        assert summary.loc["roas", "coverage_95"] == pytest.approx(0.8)
        assert summary.loc["roas", "n"] == 20

    def test_unhealthy_replications_are_excluded(self):
        results = _fake_results(1.0, n=10)
        results.loc[:4, "max_r_hat"] = 1.5  # five bad fits
        summary = summarise(results, FULL)
        assert summary.loc["roas", "n"] == 5

    def test_low_ess_replications_are_excluded(self):
        results = _fake_results(1.0, n=10)
        results.loc[:2, "min_ess_bulk"] = 5.0
        summary = summarise(results, FULL)
        assert summary.loc["roas", "n"] == 7

    def test_divergent_replications_are_excluded(self):
        """A divergent chain misses part of the posterior, so its coverage proves nothing."""
        results = _fake_results(1.0, n=10)
        results.loc[:3, "divergence_rate"] = 0.13
        summary = summarise(results, FULL)
        assert summary.loc["roas", "n"] == 6

    def test_divergence_rate_at_the_threshold_is_kept(self):
        results = _fake_results(1.0, n=10)
        results["divergence_rate"] = FULL.max_divergence_rate
        assert summarise(results, FULL).loc["roas", "n"] == 10

    def test_reports_one_row_per_parameter(self):
        results = pd.concat(
            [_fake_results(1.0, n=4, parameter="roas"), _fake_results(0.5, n=4, parameter="decay")],
            ignore_index=True,
        )
        summary = summarise(results, FULL)
        assert list(summary.index) == ["decay", "roas"]


class TestGate:
    def test_passes_when_coverage_is_near_nominal(self):
        assert check_gate(summarise(_fake_results(0.95), FULL), FULL) == []

    def test_fails_when_coverage_is_far_below_nominal(self):
        failures = check_gate(summarise(_fake_results(0.50), FULL), FULL)
        assert len(failures) == 1
        assert "coverage" in failures[0]

    def test_fails_when_every_replication_was_excluded(self):
        """Measuring nothing must not report success."""
        results = _fake_results(1.0, n=6)
        results["divergence_rate"] = 0.9
        failures = check_gate(summarise(results, FULL), FULL)
        assert len(failures) == 1
        assert "nothing was measured" in failures[0]

    def test_fails_when_coverage_is_far_above_nominal(self):
        """Needlessly wide intervals are also a failure, not a free pass."""
        profile = BenchmarkProfile(
            name="strict",
            seeds=(0,),
            n_periods=10,
            num_warmup=1,
            num_samples=1,
            coverage_tolerance=0.02,
        )
        assert check_gate(summarise(_fake_results(1.0), profile), profile) != []


class TestSimulatedExperiments:
    @pytest.fixture
    def panel(self):
        return generate_panel(DGPConfig(n_periods=300, seed=2))

    def test_no_experiments_when_no_channels_are_calibrated(self, panel):
        assert _simulate_experiments(panel, FULL, 0) == ()

    def test_one_experiment_per_calibrated_channel(self, panel):
        profile = replace(FULL, calibrated_channels=("search", "email"))
        experiments = _simulate_experiments(panel, profile, 0)
        assert [e.channel for e in experiments] == ["search", "email"]

    def test_window_length_matches_the_profile(self, panel):
        profile = replace(FULL, calibrated_channels=("search",), experiment_days=30)
        experiment = _simulate_experiments(panel, profile, 0)[0]
        assert (experiment.end - experiment.start).days + 1 == 30

    def test_standard_error_is_the_configured_fraction_of_truth(self, panel):
        profile = replace(
            FULL, calibrated_channels=("search",), experiment_days=30, experiment_relative_se=0.25
        )
        experiment = _simulate_experiments(panel, profile, 0)[0]
        index = panel.data.index
        first = int(0.4 * len(index))
        truth = float(panel.truth.contributions["search"].to_numpy()[first : first + 30].sum())
        assert experiment.standard_error == pytest.approx(0.25 * truth)

    def test_measurement_is_noisy_but_centred_on_truth(self, panel):
        """Averaged over many seeds the simulated experiment must be unbiased."""
        profile = replace(FULL, calibrated_channels=("search",), experiment_days=30)
        index = panel.data.index
        first = int(0.4 * len(index))
        truth = float(panel.truth.contributions["search"].to_numpy()[first : first + 30].sum())

        measured = [
            _simulate_experiments(panel, profile, seed)[0].incremental_revenue
            for seed in range(200)
        ]
        assert np.mean(measured) == pytest.approx(truth, rel=0.05)
        assert np.std(measured) > 0

    def test_experiment_noise_does_not_disturb_the_panel(self, panel):
        """Turning calibration on must not change the data being fitted."""
        calibrated = replace(FULL, calibrated_channels=("search",))
        again = generate_panel(DGPConfig(n_periods=300, seed=2))
        _simulate_experiments(panel, calibrated, 0)
        assert panel.data.equals(again.data)


class TestCalibrationSection:
    def test_report_gains_a_channel_group_breakdown_when_calibrated(self):
        profile = replace(FULL, calibrated_channels=("search",))
        results = _fake_results(1.0, n=6)
        results["calibrated"] = True
        report = format_report(results, summarise(results, profile), profile)
        assert "Effect of calibration, by channel group" in report
        assert "tested channels" in report

    def test_report_omits_the_breakdown_when_not_calibrated(self):
        results = _fake_results(1.0, n=6)
        report = format_report(results, summarise(results, FULL), FULL)
        assert "Effect of calibration" not in report


class TestPairedComparison:
    @pytest.fixture
    def profile(self):
        return replace(FULL, calibrated_channels=("search", "social"))

    def _arm(self, seeds, channels, bias, *, divergence=0.0):
        rows = []
        for seed in seeds:
            for channel in channels:
                rows.append(
                    {
                        "seed": seed,
                        "channel": channel,
                        "calibrated": channel in ("search", "social"),
                        "parameter": "roas",
                        "truth": 2.0,
                        "posterior_mean": 2.0 * (1 + bias),
                        "relative_bias": bias,
                        "covered_95": True,
                        "covered_50": True,
                        "interval_width_95": 1.0,
                        "max_r_hat": 1.0,
                        "min_ess_bulk": 500.0,
                        "divergences": 0,
                        "divergence_rate": divergence,
                        "seconds": 1.0,
                    }
                )
        return pd.DataFrame(rows)

    def test_pairs_only_seeds_healthy_in_both_arms(self, profile):
        baseline = self._arm([0, 1, 2], ["search"], 0.5)
        treatment = self._arm([0, 1, 2], ["search"], 0.1)
        # Seed 2 diverges in the baseline only; it must drop out of both arms.
        baseline.loc[baseline["seed"] == 2, "divergence_rate"] = 0.5

        comparison = paired_comparison(baseline, treatment, profile)

        assert comparison.attrs["common_seeds"] == [0, 1]
        assert comparison[comparison["group"] == "all"]["n"].tolist() == [2, 2]

    def test_splits_tested_from_untested_channels(self, profile):
        baseline = self._arm([0], ["search", "social", "video", "tv", "email"], 0.5)
        treatment = self._arm([0], ["search", "social", "video", "tv", "email"], 0.1)

        comparison = paired_comparison(baseline, treatment, profile)
        tested = comparison[comparison["group"] == "tested"]
        untested = comparison[comparison["group"] == "untested"]

        assert tested["n"].tolist() == [2, 2]
        assert untested["n"].tolist() == [3, 3]

    def test_raises_when_no_seed_is_healthy_in_both(self, profile):
        baseline = self._arm([0], ["search"], 0.5, divergence=0.9)
        treatment = self._arm([0], ["search"], 0.1)
        with pytest.raises(ValueError, match="no replication that is healthy in both"):
            paired_comparison(baseline, treatment, profile)

    def test_format_shows_before_and_after_and_the_paired_seed_count(self, profile):
        baseline = self._arm([0, 1], ["search", "video"], 0.5)
        treatment = self._arm([0, 1], ["search", "video"], 0.1)

        rendered = format_comparison(paired_comparison(baseline, treatment, profile), profile)

        assert "**2** replications healthy in both" in rendered
        assert "50.0% → 10.0%" in rendered
        assert "tested (search, social)" in rendered

    def test_cli_compare_writes_a_report_without_sampling(self, profile, tmp_path, monkeypatch):
        monkeypatch.setitem(PROFILES, "cmp", profile)
        baseline = tmp_path / "b.csv"
        treatment = tmp_path / "t.csv"
        self._arm([0, 1], ["search", "video"], 0.5).to_csv(baseline, index=False)
        self._arm([0, 1], ["search", "video"], 0.1).to_csv(treatment, index=False)

        exit_code = main(
            [
                "--profile",
                "cmp",
                "--compare",
                str(baseline),
                str(treatment),
                "--output",
                str(tmp_path),
            ]
        )

        assert exit_code == 0
        assert "Effect of experiment calibration" in (tmp_path / "comparison.md").read_text()


class TestMarkdownTable:
    def test_renders_header_divider_and_rows(self):
        frame = pd.DataFrame({"a": ["1", "22"], "b": ["3", "4"]}, index=["x", "y"])
        lines = _markdown_table(frame, "idx").splitlines()
        assert lines[0].startswith("| idx")
        assert set(lines[1]) <= {"|", "-"}
        assert len(lines) == 4

    def test_columns_are_width_aligned(self):
        frame = pd.DataFrame({"col": ["short", "a-much-longer-value"]}, index=["a", "b"])
        lines = _markdown_table(frame, "i").splitlines()
        assert len({len(line) for line in lines}) == 1


class TestFormatReport:
    def test_names_the_reproducing_command_and_the_profile(self):
        results = _fake_results(0.9, n=12)
        report = format_report(results, summarise(results, FULL), FULL)
        assert "just recover --profile full" in report
        assert "12 replications" in report

    def test_reports_excluded_replications(self):
        results = _fake_results(1.0, n=10)
        results.loc[:1, "max_r_hat"] = 9.0
        report = format_report(results, summarise(results, FULL), FULL)
        assert "**2** of 10" in report


class TestEndToEnd:
    def test_cli_writes_report_and_raw_results(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setitem(PROFILES, "tiny", TINY)
        exit_code = main(["--profile", "tiny", "--output", str(tmp_path)])
        assert exit_code == 0

        report = (tmp_path / "tiny.md").read_text(encoding="utf-8")
        assert "Parameter-recovery benchmark" in report
        assert "coverage 95%" in report
        # The results table must actually contain rows: an all-excluded run renders an empty
        # table while still emitting every surrounding heading.
        for parameter in ("roas", "beta", "decay", "slope", "half_saturation"):
            assert f"| {parameter}" in report

        raw = pd.read_csv(tmp_path / "tiny-raw.csv")
        assert "divergence_rate" in raw.columns
        # Five channels x four shared parameters, plus one decay per geometric channel.
        assert len(raw) == 5 * 4 + 4
        assert set(raw["parameter"]) == {"roas", "beta", "slope", "half_saturation", "decay"}
        assert np.all(np.isfinite(raw["posterior_mean"]))
        assert "Parameter-recovery benchmark" in capsys.readouterr().out

    def test_check_flag_exits_non_zero_when_the_gate_fails(self, tmp_path, monkeypatch, capsys):
        """The CI gate has to actually fail. A gate that cannot fail is decoration."""
        impossible = replace(TINY, name="impossible", coverage_tolerance=0.0)
        monkeypatch.setitem(PROFILES, "impossible", impossible)

        exit_code = main(["--profile", "impossible", "--output", str(tmp_path), "--check"])

        assert exit_code == 1
        assert "GATE FAILURES" in capsys.readouterr().err

    def test_gate_failures_are_reported_but_tolerated_without_check(
        self, tmp_path, monkeypatch, capsys
    ):
        impossible = replace(TINY, name="lenient", coverage_tolerance=0.0)
        monkeypatch.setitem(PROFILES, "lenient", impossible)

        exit_code = main(["--profile", "lenient", "--output", str(tmp_path)])

        assert exit_code == 0
        assert "GATE FAILURES" in capsys.readouterr().err
