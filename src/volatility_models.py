"""
Volatility Modeling: Realized Semivariance, Rolling Volatilities, and GARCH(1,1) Econometrics
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Compute rolling annualized realized volatility: std * sqrt(252).
    """
    return returns.rolling(window=window).std() * np.sqrt(annualization_factor)


def compute_downside_semivolatility(
    returns: pd.Series,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Compute rolling annualized downside semivolatility (volatility of negative returns).
    """
    downside_returns = returns.clip(upper=0.0)
    return downside_returns.rolling(window=window).std() * np.sqrt(annualization_factor)


def compute_parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Calculate Parkinson extreme-value volatility estimator from High/Low prices:
    sigma = sqrt( 1 / (4 * ln(2)) * sum( ln(H/L)^2 ) / N ) * sqrt(252).
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2.0))
    park_var = factor * (log_hl ** 2)
    return np.sqrt(park_var.rolling(window=window).mean() * annualization_factor)


def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "t",
    annualization_factor: int = 252
) -> Dict[str, Union[float, pd.Series, object]]:
    """
    Estimate GARCH(p,q) conditional volatility model using the arch library.

    Parameters
    ----------
    returns : pd.Series
        Asset return series (percentages or fractions; rescaled internally).
    p : int
        Lag order of symmetric innovation term (ARCH order).
    q : int
        Lag order of lagged variance (GARCH order).
    dist : str
        Error distribution ('t' for Student's t, 'normal' for Gaussian).
    annualization_factor : int
        Trading sessions per year (252).

    Returns
    -------
    dict containing:
        - 'conditional_volatility': Annualized conditional volatility series
        - 'persistence': alpha + beta persistence parameter
        - 'half_life': Volatility shock half-life in sessions
        - 'params': Fitted model parameters
        - 'fit_result': Underlying arch model fit object
    """
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("The 'arch' library is required to fit GARCH models. Install with: pip install arch") from e

    # Scale returns to percentage points for numerical convergence in optimizer
    clean_r = returns.dropna()
    scaled_r = clean_r * 100.0

    am = arch_model(scaled_r, mean="Constant", vol="GARCH", p=p, q=q, dist=dist)
    fit_res = am.fit(disp="off", show_warning=False)

    # Rescale conditional volatility back to annual standard deviation fraction
    cond_vol_annual = (fit_res.conditional_volatility / 100.0) * np.sqrt(annualization_factor)

    # Compute persistence: alpha + beta
    alpha = fit_res.params.get("alpha[1]", 0.0)
    beta = fit_res.params.get("beta[1]", 0.0)
    persistence = float(alpha + beta)

    # Half life of volatility shocks: ln(0.5) / ln(persistence)
    if 0.0 < persistence < 1.0:
        half_life = float(np.log(0.5) / np.log(persistence))
    else:
        half_life = np.nan

    return {
        "conditional_volatility": cond_vol_annual,
        "persistence": persistence,
        "half_life": half_life,
        "alpha": float(alpha),
        "beta": float(beta),
        "params": fit_res.params.to_dict(),
        "summary": fit_res.summary().as_text(),
        "fit_result": fit_res
    }
