"""
Downside Stress Testing: Deterministic Scenarios, Correlated Monte Carlo, and Sensitivity Analysis
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def run_deterministic_stress(
    weights: pd.Series,
    shocks: Union[List[float], np.ndarray] = (0.01, 0.03, 0.05, 0.10, 0.20),
    portfolio_value: float = 100_000.0
) -> pd.DataFrame:
    """
    Simulate deterministic parallel adverse shocks across portfolio assets.

    Parameters
    ----------
    weights : pd.Series
        Asset portfolio weights summing to 1.0.
    shocks : list or array of float
        Adverse shock magnitudes (e.g. 0.05 for -5% move).
    portfolio_value : float
        Notional portfolio capital base.

    Returns
    -------
    pd.DataFrame
        Table with shock magnitude, dollar loss, and percentage portfolio loss.
    """
    shock_arr = np.asarray(shocks)
    results = []
    for s in shock_arr:
        # Each holding loses fraction s
        dollar_loss = portfolio_value * (weights * s).sum()
        pct_loss = dollar_loss / portfolio_value
        results.append({
            "shock_pct": s,
            "shock_label": f"{s:.0%}",
            "dollar_loss": dollar_loss,
            "portfolio_loss_pct": pct_loss
        })
    return pd.DataFrame(results)


def run_idiosyncratic_stress(
    weights: pd.Series,
    target_ticker: str = "NVDA",
    target_shocks: Union[List[float], np.ndarray] = (0.05, 0.10, 0.15, 0.20, 0.30),
    portfolio_value: float = 100_000.0
) -> pd.DataFrame:
    """
    Simulate idiosyncratic reputational shock concentrated exclusively in the target company.
    """
    if target_ticker not in weights.index:
        raise KeyError(f"Target ticker '{target_ticker}' not present in portfolio holdings.")

    target_weight = weights[target_ticker]
    results = []
    for s in target_shocks:
        dollar_loss = portfolio_value * target_weight * s
        pct_loss = dollar_loss / portfolio_value
        results.append({
            "target_shock_pct": s,
            "target_shock_label": f"{s:.0%}",
            "target_weight": target_weight,
            "dollar_loss": dollar_loss,
            "portfolio_loss_pct": pct_loss
        })
    return pd.DataFrame(results)


def run_correlated_monte_carlo(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_sim: int = 30000,
    annual_event_prob: float = 0.02,
    shock_mean: float = -0.06,
    shock_sd: float = 0.03,
    min_shock: float = -0.50,
    seed: int = 42
) -> Dict[str, Union[float, np.ndarray, pd.DataFrame]]:
    """
    Correlated Monte Carlo scenario engine for rare reputational / corporate event shocks.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns for empirical correlation matrix.
    weights : pd.Series
        Portfolio asset weights.
    n_sim : int
        Number of simulation paths (default 30,000).
    annual_event_prob : float
        Assumed annual probability of an adverse reputational event per asset.
    shock_mean : float
        Assumed mean conditional return move given an event (-6%).
    shock_sd : float
        Assumed standard deviation of conditional shock (3%).
    min_shock : float
        Lower bound truncation for extreme downside shock (-50%).
    seed : int
        Reproducibility seed.

    Returns
    -------
    dict containing:
        - 'loss_distribution': array of simulated percentage portfolio losses
        - 'mean_loss': expected scenario loss
        - 'var_95': 95% scenario VaR
        - 'es_95': 95% scenario Expected Shortfall
        - 'var_99': 99% scenario VaR
        - 'es_99': 99% scenario Expected Shortfall
        - 'correlation_matrix': empirical correlation matrix used
    """
    assets = weights.index.tolist()
    corr_hist = returns[assets].dropna().corr().to_numpy()

    # Ensure positive semi-definiteness via eigenvalue decomposition
    eigval, eigvec = np.linalg.eigh(corr_hist)
    chol = eigvec @ np.diag(np.sqrt(np.clip(eigval, 1e-8, None)))

    rng = np.random.default_rng(seed)
    # Correlated standard normal shocks
    z = rng.standard_normal((n_sim, len(assets))) @ chol.T

    # Bernoulli event occurrences per asset
    event_hits = rng.random((n_sim, len(assets))) < annual_event_prob

    # Conditional adverse returns
    conditional_shock = np.clip(shock_mean + shock_sd * z, min_shock, 0.0)

    # Portfolio loss percentage: sum(hits * (-shock) * weight)
    w_vec = weights.to_numpy()
    loss_pct = (event_hits * (-conditional_shock) * w_vec).sum(axis=1)

    var95 = float(np.quantile(loss_pct, 0.95))
    tail95 = loss_pct[loss_pct >= var95]
    es95 = float(tail95.mean()) if len(tail95) > 0 else var95

    var99 = float(np.quantile(loss_pct, 0.99))
    tail99 = loss_pct[loss_pct >= var99]
    es99 = float(tail99.mean()) if len(tail99) > 0 else var99

    return {
        "loss_distribution": loss_pct,
        "mean_loss": float(loss_pct.mean()),
        "var_95": var95,
        "es_95": es95,
        "var_99": var99,
        "es_99": es99,
        "annual_event_prob": annual_event_prob,
        "shock_mean": shock_mean,
        "shock_sd": shock_sd,
        "n_sim": n_sim,
        "correlation_matrix": pd.DataFrame(corr_hist, index=assets, columns=assets)
    }


def run_scenario_sensitivity_matrix(
    returns: pd.DataFrame,
    weights: pd.Series,
    event_prob_grid: Union[List[float], np.ndarray] = (0.01, 0.02, 0.05),
    shock_mean_grid: Union[List[float], np.ndarray] = (-0.03, -0.06, -0.10),
    n_sim: int = 10000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Stress-testing sensitivity grid varying event probability and shock magnitude.
    """
    records = []
    for prob in event_prob_grid:
        for s_mean in shock_mean_grid:
            mc = run_correlated_monte_carlo(
                returns, weights, n_sim=n_sim, annual_event_prob=prob,
                shock_mean=s_mean, shock_sd=0.03, seed=seed
            )
            records.append({
                "annual_event_prob": f"{prob:.1%}",
                "mean_adverse_shock": f"{s_mean:.1%}",
                "mean_loss": mc["mean_loss"],
                "var_95": mc["var_95"],
                "es_95": mc["es_95"]
            })
    return pd.DataFrame(records)
