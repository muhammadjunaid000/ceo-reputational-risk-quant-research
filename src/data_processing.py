"""
Financial Data Processing, Returns Transformation, and Market Activity Diagnostics
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_returns(
    prices: pd.DataFrame,
    drop_threshold_tickers: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily arithmetic and logarithmic returns from price series.

    Parameters
    ----------
    prices : pd.DataFrame
        Asset adjusted closing prices.
    drop_threshold_tickers : list of str, optional
        Tickers required to be non-null for return validity.

    Returns
    -------
    returns : pd.DataFrame
        Simple arithmetic daily returns.
    log_returns : pd.DataFrame
        Continuous logarithmic daily returns.
    """
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    log_returns = np.log(prices).diff().replace([np.inf, -np.inf], np.nan)

    if drop_threshold_tickers:
        valid_tickers = [t for t in drop_threshold_tickers if t in returns.columns]
        if valid_tickers:
            returns = returns.dropna(subset=valid_tickers)
            log_returns = log_returns.loc[returns.index]

    return returns, log_returns


def compute_indexed_wealth(
    returns: pd.DataFrame,
    initial_wealth: float = 100.0
) -> pd.DataFrame:
    """
    Compute cumulative wealth indexed to initial_wealth (default 100.0).
    """
    return initial_wealth * (1.0 + returns.fillna(0.0)).cumprod()


def compute_volume_surprise(
    volume: pd.Series,
    window: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate volume surprise Z-score using trailing window statistics.
    Uses shift(1) to strictly eliminate look-ahead bias.

    Parameters
    ----------
    volume : pd.Series
        Historical daily trading volume.
    window : int
        Trailing session lookback window (default 20).

    Returns
    -------
    volume_z : pd.Series
        Standardized volume surprise Z-score.
    rolling_mean : pd.Series
        Trailing baseline mean volume.
    rolling_std : pd.Series
        Trailing baseline volume standard deviation.
    """
    v_clean = volume.astype(float)
    rolling_mean = v_clean.rolling(window=window).mean().shift(1)
    rolling_std = v_clean.rolling(window=window).std().shift(1)
    volume_z = (v_clean - rolling_mean) / rolling_std.replace(0, np.nan)
    return volume_z, rolling_mean, rolling_std


def compute_dollar_turnover(
    prices: pd.DataFrame,
    volume: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate daily dollar turnover (Price * Volume) for available assets.
    """
    common_cols = [c for c in prices.columns if c in volume.columns]
    return prices[common_cols] * volume[common_cols]


def compute_rolling_beta(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Compute rolling market beta: Cov(R_i, R_m) / Var(R_m).
    """
    rolling_cov = asset_returns.rolling(window=window).cov(market_returns)
    rolling_var = market_returns.rolling(window=window).var()
    return rolling_cov / rolling_var.replace(0, np.nan)


def generate_cross_asset_summary(
    returns: pd.DataFrame,
    drawdowns: pd.DataFrame,
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Compile comprehensive risk-return summary table across all assets.
    """
    ann_ret = returns.mean() * annualization_factor
    ann_vol = returns.std() * np.sqrt(annualization_factor)
    ret_vol_ratio = ann_ret / ann_vol.replace(0, np.nan)

    summary = pd.DataFrame({
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "return_vol_ratio": ret_vol_ratio,
        "worst_day": returns.min(),
        "daily_5pct_quantile": returns.quantile(0.05),
        "max_drawdown": drawdowns.min()
    }).sort_values("annualized_volatility", ascending=False)

    return summary

