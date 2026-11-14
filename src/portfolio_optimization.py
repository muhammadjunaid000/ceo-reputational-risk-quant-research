"""
Portfolio Construction, Risk Attribution, Euler Decomposition, and Markowitz Optimization
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def validate_portfolio_weights(
    weights: Union[pd.Series, Dict[str, float]],
    tolerance: float = 1e-5
) -> pd.Series:
    """
    Validate that portfolio weights sum to 1.0 and exclude zero-weight assets.

    Returns
    -------
    pd.Series
        Cleaned, positive-weight portfolio series.
    """
    w = pd.Series(weights).astype(float)
    # Filter out zero or negative weights
    active_w = w[w > tolerance].copy()

    total_weight = active_w.sum()
    if not np.isclose(total_weight, 1.0, atol=tolerance):
        raise ValueError(
            f"Portfolio weights must sum to 1.0 (actual sum: {total_weight:.6f})."
        )
    return active_w


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series
) -> pd.Series:
    """
    Compute daily portfolio return series: R_p = sum(w_i * R_i).
    """
    valid_assets = [t for t in weights.index if t in returns.columns]
    if len(valid_assets) < len(weights):
        missing = set(weights.index) - set(valid_assets)
        raise KeyError(f"Portfolio assets missing from returns data: {missing}")

    w_aligned = weights.loc[valid_assets]
    return (returns[valid_assets] * w_aligned).sum(axis=1)


def compute_euler_risk_contribution(
    returns: pd.DataFrame,
    weights: pd.Series,
    annualization_factor: int = 252
) -> pd.DataFrame:
    """
    Calculate Euler risk decomposition:
    - Marginal Contribution to Risk (MCR): d(sigma_p) / d(w_i) = (Sigma w)_i / sigma_p
    - Component Risk Contribution (CR): w_i * MCR_i
    - Percentage Contribution to Risk (PCR): CR_i / sigma_p = w_i * (Sigma w)_i / sigma_p^2

    Guarantees sum(PCR) == 100%.

    Returns
    -------
    pd.DataFrame
        Table with weight, marginal risk, absolute risk, and percentage risk contribution.
    """
    assets = weights.index.tolist()
    cov_annual = returns[assets].cov() * annualization_factor
    w_vec = weights.to_numpy()

    port_variance = float(w_vec.T @ cov_annual.to_numpy() @ w_vec)
    port_sigma = np.sqrt(port_variance)

    # Marginal risk contribution vector
    sigma_w = cov_annual.to_numpy() @ w_vec
    mcr = sigma_w / port_sigma

    # Absolute risk contribution
    cr = w_vec * mcr

    # Percentage risk contribution
    pcr = cr / port_sigma

    df = pd.DataFrame({
        "weight": weights,
        "marginal_risk_contribution": mcr,
        "absolute_risk_contribution": cr,
        "percentage_risk_contribution": pcr,
        "risk_to_weight_ratio": pcr / weights
    }, index=weights.index)

    return df


def calculate_portfolio_performance_metrics(
    portfolio_returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> Dict[str, float]:
    """
    Calculate annualized return, annualized volatility, Sharpe ratio, and total sample return.
    """
    cum_wealth = (1.0 + portfolio_returns).cumprod()
    total_return = float(cum_wealth.iloc[-1] - 1.0) if len(cum_wealth) > 0 else 0.0

    ann_return = float(portfolio_returns.mean() * annualization_factor)
    ann_vol = float(portfolio_returns.std() * np.sqrt(annualization_factor))
    sharpe = float((ann_return - risk_free_rate) / ann_vol) if ann_vol > 0 else np.nan

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe
    }


def compute_monthly_returns(portfolio_returns: pd.Series) -> pd.DataFrame:
    """
    Convert daily portfolio returns into Year x Month table of percentage returns.
    """
    monthly = portfolio_returns.groupby([portfolio_returns.index.year, portfolio_returns.index.month]).apply(
        lambda r: (1.0 + r).prod() - 1.0
    )
    monthly.index.names = ["Year", "Month"]
    monthly_table = monthly.unstack(level="Month") * 100.0

    # Month column names
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }
    monthly_table.rename(columns=month_names, inplace=True)
    return monthly_table


def generate_markowitz_efficient_frontier(
    returns: pd.DataFrame,
    assets: List[str],
    num_portfolios: int = 4000,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
    seed: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Generate Markowitz Efficient Frontier via Monte Carlo simulation and analytical boundary.

    Returns
    -------
    sim_df : pd.DataFrame
        Simulated portfolios with returns, volatilities, and Sharpe ratios.
    min_vol_port : dict
        Minimum variance portfolio weights, return, and volatility.
    max_sharpe_port : dict
        Maximum Sharpe ratio (tangency) portfolio weights, return, and volatility.
    """
    rng = np.random.default_rng(seed)
    asset_ret = returns[assets].dropna()
    mean_ret = asset_ret.mean().to_numpy() * annualization_factor
    cov_mat = asset_ret.cov().to_numpy() * annualization_factor
    n_assets = len(assets)

    # 1. Monte Carlo simulation of random portfolios
    port_vols = np.empty(num_portfolios)
    port_rets = np.empty(num_portfolios)
    weights_record = np.empty((num_portfolios, n_assets))

    for i in range(num_portfolios):
        raw_w = rng.random(n_assets)
        w = raw_w / np.sum(raw_w)
        weights_record[i, :] = w
        r_p = float(w @ mean_ret)
        v_p = float(np.sqrt(w.T @ cov_mat @ w))
        port_rets[i] = r_p
        port_vols[i] = v_p

    sharpe_ratios = (port_rets - risk_free_rate) / port_vols
    sim_df = pd.DataFrame({
        "return": port_rets,
        "volatility": port_vols,
        "sharpe": sharpe_ratios
    })

    # 2. Optimization for Minimum Volatility Portfolio
    def port_vol_obj(w):
        return np.sqrt(w.T @ cov_mat @ w)

    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    init_w = np.ones(n_assets) / n_assets

    res_min_vol = minimize(port_vol_obj, init_w, method="SLSQP", bounds=bounds, constraints=eq_constraint)
    min_vol_weights = pd.Series(res_min_vol.x, index=assets)
    min_vol_metrics = {
        "weights": min_vol_weights.to_dict(),
        "return": float(min_vol_weights.to_numpy() @ mean_ret),
        "volatility": float(res_min_vol.fun),
        "sharpe": float((min_vol_weights.to_numpy() @ mean_ret - risk_free_rate) / res_min_vol.fun)
    }

    # 3. Optimization for Maximum Sharpe Portfolio
    def neg_sharpe_obj(w):
        r = w @ mean_ret
        v = np.sqrt(w.T @ cov_mat @ w)
        return -(r - risk_free_rate) / v if v > 0 else 0.0

    res_max_sharpe = minimize(neg_sharpe_obj, init_w, method="SLSQP", bounds=bounds, constraints=eq_constraint)
    max_sharpe_weights = pd.Series(res_max_sharpe.x, index=assets)
    ms_vol = float(np.sqrt(max_sharpe_weights.to_numpy() @ cov_mat @ max_sharpe_weights.to_numpy()))
    ms_ret = float(max_sharpe_weights.to_numpy() @ mean_ret)
    max_sharpe_metrics = {
        "weights": max_sharpe_weights.to_dict(),
        "return": ms_ret,
        "volatility": ms_vol,
        "sharpe": float((ms_ret - risk_free_rate) / ms_vol)
    }

    return sim_df, min_vol_metrics, max_sharpe_metrics
