"""
Comprehensive Risk Analytics: Downside Metrics, VaR, Expected Shortfall, and Drawdown Decomposition
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_drawdown_series(returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Compute cumulative wealth and percentage drawdown from previous peak.

    Returns
    -------
    wealth : pd.Series
        Cumulative wealth index (starting at 1.0).
    drawdowns : pd.Series
        Drawdown series in fractions (<= 0.0).
    """
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    cummax = wealth.cummax()
    drawdowns = (wealth - cummax) / cummax
    return wealth, drawdowns


def identify_drawdown_episodes(
    wealth: pd.Series,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Isolate distinct, non-overlapping drawdown cycles.
    Distinguishes separate economic drawdown episodes from repeated threshold crossings.

    A distinct drawdown episode begins when wealth drops below its previous peak (DD < 0)
    and strictly terminates only when a new peak is achieved (DD == 0) or at sample end.

    Parameters
    ----------
    wealth : pd.Series
        Cumulative wealth index.
    top_n : int
        Number of deepest episodes to return.

    Returns
    -------
    pd.DataFrame
        Table of distinct drawdown episodes sorted by severity.
    """
    cummax = wealth.cummax()
    drawdown = (wealth - cummax) / cummax

    episodes = []
    in_drawdown = False
    cur_peak = wealth.index[0]
    cur_trough = cur_peak
    cur_min_dd = 0.0

    for date, val in wealth.items():
        cur_dd = drawdown.loc[date]
        if cur_dd < 0:
            if not in_drawdown:
                in_drawdown = True
                cur_peak = wealth.loc[:date].idxmax()
                cur_trough = date
                cur_min_dd = cur_dd
            else:
                if cur_dd < cur_min_dd:
                    cur_min_dd = cur_dd
                    cur_trough = date
        else:
            if in_drawdown:
                # Episode recovered
                peak_to_trough = len(wealth.loc[cur_peak:cur_trough]) - 1
                trough_to_recovery = len(wealth.loc[cur_trough:date]) - 1
                total_duration = len(wealth.loc[cur_peak:date]) - 1
                episodes.append({
                    "peak_date": cur_peak,
                    "trough_date": cur_trough,
                    "recovery_date": date,
                    "max_drawdown": cur_min_dd,
                    "peak_to_trough_sessions": peak_to_trough,
                    "recovery_sessions": trough_to_recovery,
                    "total_duration_sessions": total_duration,
                    "status": "Recovered"
                })
                in_drawdown = False

    if in_drawdown:
        # Unrecovered ongoing drawdown at end of series
        peak_to_trough = len(wealth.loc[cur_peak:cur_trough]) - 1
        total_duration = len(wealth.loc[cur_peak:]) - 1
        episodes.append({
            "peak_date": cur_peak,
            "trough_date": cur_trough,
            "recovery_date": pd.NaT,
            "max_drawdown": cur_min_dd,
            "peak_to_trough_sessions": peak_to_trough,
            "recovery_sessions": np.nan,
            "total_duration_sessions": total_duration,
            "status": "Unrecovered"
        })

    if not episodes:
        return pd.DataFrame()

    df = pd.DataFrame(episodes).sort_values("max_drawdown", ascending=True).reset_index(drop=True)
    return df.head(top_n)


def compute_var_and_es(
    returns: pd.Series,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Calculate 1-day Value-at-Risk (VaR) and Expected Shortfall (CVaR) using
    Historical, Parametric (Gaussian), and Cornish-Fisher methodologies.

    Reported as positive loss numbers (e.g. 0.05 means 5% loss).

    Parameters
    ----------
    returns : pd.Series
        Asset or portfolio return series.
    alpha : float
        Tail probability (0.05 for 95% confidence, 0.01 for 99% confidence).

    Returns
    -------
    dict of risk metrics.
    """
    clean_r = returns.dropna()
    mu = float(clean_r.mean())
    sigma = float(clean_r.std())
    skew = float(clean_r.skew())
    kurt = float(clean_r.kurtosis())  # excess kurtosis

    # 1. Historical VaR and ES
    hist_quantile = float(clean_r.quantile(alpha))
    hist_var = -hist_quantile
    tail_losses = clean_r[clean_r <= hist_quantile]
    hist_es = float(-tail_losses.mean()) if len(tail_losses) > 0 else hist_var

    # 2. Parametric (Gaussian) VaR and ES
    z_alpha = float(norm.ppf(1.0 - alpha))
    param_var = -(mu - z_alpha * sigma)
    # ES = -mu + sigma * phi(z_alpha) / alpha
    param_es = -(mu - sigma * norm.pdf(z_alpha) / alpha)

    # 3. Cornish-Fisher (Modified) VaR
    # Accounts for skewness and excess kurtosis
    z_cf = (
        z_alpha
        + (z_alpha**2 - 1.0) * skew / 6.0
        + (z_alpha**3 - 3.0 * z_alpha) * kurt / 24.0
        - (2.0 * z_alpha**3 - 5.0 * z_alpha) * (skew**2) / 36.0
    )
    cf_var = -(mu - z_cf * sigma)

    return {
        "alpha": alpha,
        "confidence_level": 1.0 - alpha,
        "historical_var": hist_var,
        "historical_es": hist_es,
        "parametric_var": param_var,
        "parametric_es": param_es,
        "cornish_fisher_var": cf_var,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "mean_daily_return": mu,
        "daily_volatility": sigma
    }


def compute_rolling_var_es(
    returns: pd.Series,
    window: int = 60,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute rolling historical and parametric 95% VaR and Expected Shortfall over trailing window.
    """
    clean_r = returns.dropna()
    rolling_hist_var = -clean_r.rolling(window=window).quantile(alpha)
    rolling_mean = clean_r.rolling(window=window).mean()
    rolling_std = clean_r.rolling(window=window).std()

    z_alpha = norm.ppf(1.0 - alpha)
    rolling_param_var = -(rolling_mean - z_alpha * rolling_std)
    rolling_param_es = -(rolling_mean - rolling_std * norm.pdf(z_alpha) / alpha)

    # Historical rolling ES
    def calc_hist_es(x):
        q = np.quantile(x, alpha)
        tail = x[x <= q]
        return -np.mean(tail) if len(tail) > 0 else -q

    rolling_hist_es = clean_r.rolling(window=window).apply(calc_hist_es, raw=True)

    return pd.DataFrame({
        "historical_var": rolling_hist_var,
        "historical_es": rolling_hist_es,
        "parametric_var": rolling_param_var,
        "parametric_es": rolling_param_es
    }, index=clean_r.index)

