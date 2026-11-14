"""
Unit Tests for Drawdown Decomposition, Recovery Cycles, and Episode Segmentation
"""

import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import compute_drawdown_series, identify_drawdown_episodes


def test_drawdown_series_properties():
    """Verify basic properties of cumulative wealth and underwater series."""
    returns = pd.Series([0.05, -0.02, -0.03, 0.04, -0.01])
    wealth, dd = compute_drawdown_series(returns)

    assert (wealth >= 0).all()
    assert (dd <= 0.0).all()
    # At day 0 (5% gain), wealth is 1.05 and drawdown is 0
    assert np.isclose(dd.iloc[0], 0.0)


def test_distinct_episodes_vs_threshold_crossings():
    """
    CRITICAL METHODOLOGICAL TEST:
    Distinguish a single economic drawdown cycle with oscillations from multiple threshold crossings.

    Scenario:
    - Day 0: Peak at 100
    - Day 1: Drops to 92 (-8% DD) -> Crosses -5% threshold
    - Day 2: Rebounds to 96 (-4% DD) -> Rises above -5% threshold
    - Day 3: Drops to 88 (-12% DD) -> Crosses -5% threshold again, reaches trough
    - Day 4: Rebounds to 96 (-4% DD)
    - Day 5: Reaches 102 (+2% new peak) -> Full recovery

    A naive threshold counter sees two '-5%' events.
    The economic episode engine must correctly identify exactly ONE single drawdown episode
    with peak on Day 0, trough on Day 3 (-12%), and recovery on Day 5.
    """
    dates = pd.date_range("2025-01-01", periods=6, freq="B")
    # Prices: 100, 92, 96, 88, 96, 102
    prices = pd.Series([100.0, 92.0, 96.0, 88.0, 96.0, 102.0], index=dates)
    wealth = prices / prices.iloc[0]

    episodes = identify_drawdown_episodes(wealth, top_n=5)

    assert len(episodes) == 1, "Must recognize exactly ONE drawdown episode despite multiple threshold oscillations."
    ep = episodes.iloc[0]
    assert ep["peak_date"] == dates[0]
    assert ep["trough_date"] == dates[3]
    assert ep["recovery_date"] == dates[5]
    assert np.isclose(ep["max_drawdown"], -0.12, atol=1e-5)
    assert ep["status"] == "Recovered"
    assert ep["total_duration_sessions"] == 5


def test_unrecovered_drawdown():
    """Verify handling of ongoing drawdown at end of historical sample."""
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    # Prices: 100, 110 (peak), 105, 95 (trough), 98
    prices = pd.Series([100.0, 110.0, 105.0, 95.0, 98.0], index=dates)
    returns = prices.pct_change().dropna()

    wealth, dd = compute_drawdown_series(returns)
    episodes = identify_drawdown_episodes(wealth, top_n=5)

    assert len(episodes) == 1
    ep = episodes.iloc[0]
    assert ep["peak_date"] == dates[1]
    assert ep["trough_date"] == dates[3]
    assert pd.isna(ep["recovery_date"])
    assert ep["status"] == "Unrecovered"
    assert np.isclose(ep["max_drawdown"], (95.0 - 110.0) / 110.0, atol=1e-5)
