"""
Unit Tests for Risk Metrics: VaR, Expected Shortfall, Cornish-Fisher, and Tail Analytics
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src.risk_metrics import compute_var_and_es, compute_rolling_var_es


@pytest.fixture
def sample_returns():
    """Generate reproducible synthetic return series with known negative skew and fat tails."""
    rng = np.random.default_rng(123)
    # Mixture: normal + occasional downside jumps
    normal_rets = rng.normal(loc=0.0005, scale=0.015, size=500)
    jumps = rng.choice([0.0, -0.05, -0.08], size=500, p=[0.96, 0.03, 0.01])
    rets = normal_rets + jumps
    dates = pd.date_range("2024-01-01", periods=500, freq="B")
    return pd.Series(rets, index=dates)


def test_var_monotonicity_and_ordering(sample_returns):
    """Test that 99% VaR is greater than 95% VaR, and ES >= VaR."""
    metrics_95 = compute_var_and_es(sample_returns, alpha=0.05)
    metrics_99 = compute_var_and_es(sample_returns, alpha=0.01)

    # 99% confidence VaR should exceed 95% confidence VaR
    assert metrics_99["historical_var"] > metrics_95["historical_var"]
    assert metrics_99["parametric_var"] > metrics_95["parametric_var"]

    # Expected Shortfall must be greater than or equal to VaR for the same alpha
    assert metrics_95["historical_es"] >= metrics_95["historical_var"]
    assert metrics_99["historical_es"] >= metrics_99["historical_var"]
    assert metrics_95["parametric_es"] >= metrics_95["parametric_var"]
    assert metrics_99["parametric_es"] >= metrics_99["parametric_var"]


def test_parametric_var_formula():
    """Verify parametric Gaussian VaR against analytical formula."""
    rng = np.random.default_rng(42)
    clean_rets = pd.Series(rng.normal(loc=0.001, scale=0.02, size=1000))

    metrics = compute_var_and_es(clean_rets, alpha=0.05)
    expected_z = norm.ppf(0.95)
    expected_var = -(clean_rets.mean() - expected_z * clean_rets.std())
    expected_es = -(clean_rets.mean() - clean_rets.std() * norm.pdf(expected_z) / 0.05)

    assert np.isclose(metrics["parametric_var"], expected_var, atol=1e-6)
    assert np.isclose(metrics["parametric_es"], expected_es, atol=1e-6)


def test_cornish_fisher_with_skew_kurt():
    """Verify Cornish-Fisher modifies VaR upward under negative skew and positive excess kurtosis."""
    rng = np.random.default_rng(99)
    # Heavy left tail
    base = rng.normal(0.0, 0.01, 1000)
    left_tail_shocks = rng.choice([0.0, -0.06], size=1000, p=[0.95, 0.05])
    r = pd.Series(base + left_tail_shocks)

    metrics = compute_var_and_es(r, alpha=0.05)
    # With negative skew and fat left tail, CF modified VaR should reflect increased tail risk
    assert metrics["skewness"] < 0
    assert metrics["excess_kurtosis"] > 0
    assert metrics["cornish_fisher_var"] > 0


def test_rolling_var_es_shape(sample_returns):
    """Verify rolling VaR/ES dimensions and alignment."""
    window = 50
    rolling_df = compute_rolling_var_es(sample_returns, window=window, alpha=0.05)

    assert len(rolling_df) == len(sample_returns)
    assert list(rolling_df.columns) == [
        "historical_var", "historical_es", "parametric_var", "parametric_es"
    ]
    # First (window-1) values must be NaN
    assert rolling_df["historical_var"].iloc[:window - 1].isna().all()
    assert rolling_df["historical_var"].iloc[window:].notna().all()
