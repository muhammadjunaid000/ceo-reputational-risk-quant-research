"""
Unit Tests for Portfolio Allocation, Weights Validation, and Euler Risk Decomposition
"""

import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimization import (
    validate_portfolio_weights, compute_portfolio_returns,
    compute_euler_risk_contribution, calculate_portfolio_performance_metrics
)


@pytest.fixture
def multi_asset_returns():
    """Create a 5-asset return DataFrame for testing."""
    rng = np.random.default_rng(42)
    assets = ["NVDA", "META", "AMZN", "TSLA", "MSFT"]
    # Correlated returns
    cov = np.array([
        [0.0009, 0.0004, 0.0003, 0.0005, 0.0002],
        [0.0004, 0.0007, 0.0003, 0.0004, 0.0002],
        [0.0003, 0.0003, 0.0006, 0.0003, 0.0002],
        [0.0005, 0.0004, 0.0003, 0.0012, 0.0003],
        [0.0002, 0.0002, 0.0002, 0.0003, 0.0004]
    ])
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((250, 5))
    r = z @ L.T
    dates = pd.date_range("2025-01-01", periods=250, freq="B")
    return pd.DataFrame(r, index=dates, columns=assets)


def test_weights_validation():
    """Ensure weights sum to 1.0, throw error if not, and filter zero weights."""
    valid_weights = {"NVDA": 0.30, "META": 0.20, "AMZN": 0.20, "TSLA": 0.15, "MSFT": 0.15}
    w_clean = validate_portfolio_weights(valid_weights)
    assert np.isclose(w_clean.sum(), 1.0)
    assert len(w_clean) == 5

    # Zero weights should be pruned
    with_zeros = {"NVDA": 0.30, "META": 0.20, "AMZN": 0.20, "TSLA": 0.15, "MSFT": 0.15, "GOOGL": 0.0}
    w_pruned = validate_portfolio_weights(with_zeros)
    assert "GOOGL" not in w_pruned.index
    assert len(w_pruned) == 5

    # Invalid weights sum
    invalid_weights = {"NVDA": 0.40, "META": 0.20}
    with pytest.raises(ValueError, match="Portfolio weights must sum to 1.0"):
        validate_portfolio_weights(invalid_weights)


def test_euler_risk_contribution_sum(multi_asset_returns):
    """
    Test Euler's theorem on homogeneous functions:
    Component percentage risk contributions must sum exactly to 100%.
    """
    weights = validate_portfolio_weights({
        "NVDA": 0.30, "META": 0.20, "AMZN": 0.20, "TSLA": 0.15, "MSFT": 0.15
    })
    risk_df = compute_euler_risk_contribution(multi_asset_returns, weights)

    # Sum of percentage risk contributions must equal 1.0
    pcr_sum = risk_df["percentage_risk_contribution"].sum()
    assert np.isclose(pcr_sum, 1.0, atol=1e-5)

    # Weights do not equal risk contributions in general
    weights_vec = risk_df["weight"].to_numpy()
    pcr_vec = risk_df["percentage_risk_contribution"].to_numpy()
    assert not np.allclose(weights_vec, pcr_vec, atol=1e-3)


def test_portfolio_returns_calculation(multi_asset_returns):
    """Verify linear combination of portfolio returns."""
    weights = pd.Series({"NVDA": 0.5, "META": 0.5})
    port_r = compute_portfolio_returns(multi_asset_returns, weights)

    expected = 0.5 * multi_asset_returns["NVDA"] + 0.5 * multi_asset_returns["META"]
    assert np.allclose(port_r, expected)


def test_performance_metrics_calculation(multi_asset_returns):
    """Verify annualized return and annualized volatility formulas."""
    weights = validate_portfolio_weights({
        "NVDA": 0.30, "META": 0.20, "AMZN": 0.20, "TSLA": 0.15, "MSFT": 0.15
    })
    port_r = compute_portfolio_returns(multi_asset_returns, weights)
    metrics = calculate_portfolio_performance_metrics(port_r, risk_free_rate=0.0)

    expected_ann_ret = port_r.mean() * 252
    expected_ann_vol = port_r.std() * np.sqrt(252)

    assert np.isclose(metrics["annualized_return"], expected_ann_ret)
    assert np.isclose(metrics["annualized_volatility"], expected_ann_vol)

