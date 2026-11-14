"""
Event Study Engine: Abnormal Returns, Estimation Models, Bootstrap & Placebo Diagnostics
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def map_to_session(calendar: pd.DatetimeIndex, date: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """
    Map an announcement date or timestamp to the next available trading session.
    """
    ts = pd.Timestamp(date)
    pos = calendar.searchsorted(ts)
    return calendar[pos] if pos < len(calendar) else pd.NaT


def build_peer_control_series(
    returns: pd.DataFrame,
    target: str = "NVDA",
    peer_pool: Optional[List[str]] = None
) -> pd.Series:
    """
    Construct an equal-weighted peer control benchmark excluding the target asset.
    """
    if peer_pool is None:
        peer_pool = [c for c in returns.columns if c != target and not c.startswith("^")]
    active_peers = [t for t in peer_pool if t != target and t in returns.columns]
    if len(active_peers) < 2:
        raise ValueError("Insufficient peer pool to construct peer-control benchmark (minimum 2 required).")
    return returns[active_peers].mean(axis=1).rename("PEER_EQW")


def get_factor_matrix(
    returns: pd.DataFrame,
    spec: str = "market",
    target: str = "NVDA",
    market: str = "SPY",
    sector: str = "SOXX",
    peer_pool: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Assemble the explanatory factor matrix for the specified model.
    """
    if spec == "market":
        if market not in returns.columns:
            raise KeyError(f"Market benchmark '{market}' not found in returns.")
        return returns[[market]].copy()
    elif spec == "market_sector":
        cols = [c for c in [market, sector] if c in returns.columns]
        if len(cols) < 2:
            raise KeyError(f"Both '{market}' and '{sector}' must be present for market_sector model.")
        return returns[cols].copy()
    elif spec == "peer_control":
        peer_series = build_peer_control_series(returns, target=target, peer_pool=peer_pool)
        return peer_series.to_frame()
    else:
        raise ValueError(f"Unknown expected-return model specification: '{spec}'")


def fit_event_model(
    returns: pd.DataFrame,
    target: str,
    day0: pd.Timestamp,
    spec: str = "market",
    estimation_length: int = 120,
    estimation_gap: int = 10,
    market: str = "SPY",
    sector: str = "SOXX",
    peer_pool: Optional[List[str]] = None
):
    """
    Fit the OLS regression model on the pre-event estimation window.
    """
    pos = returns.index.searchsorted(day0)
    est_end = pos - estimation_gap
    est_start = est_end - estimation_length

    if est_start < 0:
        est_start = 0

    est_slice = returns.iloc[est_start:est_end]
    factors = get_factor_matrix(
        returns, spec=spec, target=target, market=market, sector=sector, peer_pool=peer_pool
    )

    fit_df = pd.concat([returns[target].rename("y"), factors], axis=1).dropna()
    fit_df = fit_df.loc[est_slice.index.intersection(fit_df.index)]

    min_obs = max(40, 15 * fit_df.shape[1])
    if len(fit_df) < min_obs:
        raise ValueError(
            f"Insufficient observations in estimation window ({len(fit_df)} obs, required {min_obs})."
        )

    X = sm.add_constant(fit_df.drop(columns="y"))
    y = fit_df["y"]
    model = sm.OLS(y, X).fit()
    return model, factors


def compute_abnormal_returns(
    returns: pd.DataFrame,
    target: str,
    day0: pd.Timestamp,
    spec: str = "market",
    pre_window: int = 10,
    post_window: int = 10,
    estimation_length: int = 120,
    estimation_gap: int = 10,
    market: str = "SPY",
    sector: str = "SOXX",
    peer_pool: Optional[List[str]] = None
) -> Tuple[pd.Series, sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame, pd.DatetimeIndex]:
    """
    Compute daily abnormal returns (AR) over the specified event window.
    """
    model, factors = fit_event_model(
        returns, target=target, day0=day0, spec=spec,
        estimation_length=estimation_length, estimation_gap=estimation_gap,
        market=market, sector=sector, peer_pool=peer_pool
    )

    pos = returns.index.searchsorted(day0)
    lo = max(0, pos - pre_window)
    hi = min(len(returns), pos + post_window + 1)
    dates = returns.index[lo:hi]

    factor_slice = factors.loc[dates]
    X_pred = sm.add_constant(factor_slice, has_constant="add").reindex(columns=model.params.index)
    expected_ret = X_pred @ model.params
    ar = returns.loc[dates, target] - expected_ret

    rel_sessions = np.arange(lo - pos, hi - pos)
    ar.index = rel_sessions
    return ar, model, factor_slice, dates


def block_bootstrap_p_value(
    residuals: np.ndarray,
    observed_car: float,
    window_length: int,
    reps: int = 3000,
    block_length: int = 3,
    seed: int = 42
) -> float:
    """
    Calculate moving block bootstrap p-value under the null hypothesis of zero abnormal returns.
    """
    resid = np.asarray(residuals).dropna() if hasattr(residuals, "dropna") else np.asarray(residuals)
    resid = resid[~np.isnan(resid)]
    n_resid = len(resid)

    if n_resid < block_length + 1 or window_length <= 0:
        return np.nan

    rng = np.random.default_rng(seed)
    sims = np.empty(reps)

    for b in range(reps):
        sampled_blocks = []
        while len(sampled_blocks) < window_length:
            idx = rng.integers(0, n_resid - block_length + 1)
            sampled_blocks.extend(resid[idx:idx + block_length])
        sims[b] = np.sum(sampled_blocks[:window_length])

    p_val = (1.0 + np.sum(np.abs(sims) >= np.abs(observed_car))) / (reps + 1.0)
    return float(p_val)


def run_event_analysis(
    returns: pd.DataFrame,
    events: pd.DataFrame,
    target: str = "NVDA",
    specs: Optional[List[str]] = None,
    windows: Optional[List[Tuple[int, int]]] = None,
    estimation_length: int = 120,
    estimation_gap: int = 10,
    bootstrap_reps: int = 3000
) -> pd.DataFrame:
    """
    Full event-study pipeline across verified events, benchmark specifications, and windows.
    Returns structured results dataframe with Benjamini-Hochberg FDR correction.
    """
    if specs is None:
        specs = ["market", "peer_control", "market_sector"]
    if windows is None:
        windows = [(-1, 1), (-3, 3), (-5, 5), (0, 0)]

    # Filter for verified and analysis-ready events
    events_copy = events.copy()
    if "source_present" not in events_copy.columns:
        events_copy["source_present"] = events_copy["source_url"].fillna("").str.startswith(("https://", "http://"))
    if "analysis_ready" not in events_copy.columns:
        events_copy["analysis_ready"] = (
            events_copy["verified"].astype(bool) &
            events_copy["include"].astype(bool) &
            events_copy["source_present"]
        )

    active = events_copy[events_copy["analysis_ready"]].copy()
    if active.empty:
        # Return empty structured dataframe documenting no enabled verified events
        cols = [
            "event_id", "model", "window", "days_used", "complete_window",
            "CAR", "SE", "t_stat", "ci_low", "ci_high", "p_value", "p_adj_BH",
            "bootstrap_p", "estimation_R2"
        ]
        return pd.DataFrame(columns=cols)

    results = []
    for _, ev in active.iterrows():
        day0_mapped = map_to_session(returns.index, ev["day0"])
        if pd.isna(day0_mapped):
            continue

        for spec in specs:
            try:
                max_pre = max(abs(w[0]) for w in windows)
                max_post = max(w[1] for w in windows)
                ar, model, _, _ = compute_abnormal_returns(
                    returns, target=target, day0=day0_mapped, spec=spec,
                    pre_window=max_pre, post_window=max_post,
                    estimation_length=estimation_length, estimation_gap=estimation_gap
                )

                for w in windows:
                    mask = (ar.index >= w[0]) & (ar.index <= w[1])
                    sub_ar = ar.loc[mask]
                    L_expected = w[1] - w[0] + 1
                    L_actual = len(sub_ar)
                    car = float(sub_ar.sum()) if L_actual > 0 else np.nan
                    se = float(np.sqrt(L_actual * model.mse_resid)) if L_actual > 0 else np.nan
                    t_stat = car / se if (se and not np.isnan(se) and se > 0) else np.nan
                    p_val = 2.0 * (1.0 - st.t.cdf(abs(t_stat), df=model.df_resid)) if not np.isnan(t_stat) else np.nan
                    boot_p = block_bootstrap_p_value(
                        model.resid, car, window_length=L_actual, reps=bootstrap_reps
                    ) if L_actual > 0 else np.nan

                    results.append({
                        "event_id": ev["event_id"],
                        "model": spec,
                        "window": f"[{w[0]:+d},{w[1]:+d}]",
                        "days_used": L_actual,
                        "complete_window": L_actual == L_expected,
                        "CAR": car,
                        "SE": se,
                        "t_stat": t_stat,
                        "ci_low": car - 1.96 * se if not np.isnan(se) else np.nan,
                        "ci_high": car + 1.96 * se if not np.isnan(se) else np.nan,
                        "p_value": p_val,
                        "bootstrap_p": boot_p,
                        "estimation_R2": float(model.rsquared)
                    })
            except Exception:
                continue

    if not results:
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    valid_p = res_df["p_value"].notna()
    res_df["p_adj_BH"] = np.nan
    if valid_p.any():
        res_df.loc[valid_p, "p_adj_BH"] = multipletests(res_df.loc[valid_p, "p_value"], method="fdr_bh")[1]

    return res_df


def generate_placebo_distribution(
    returns: pd.DataFrame,
    day0: pd.Timestamp,
    target: str = "NVDA",
    spec: str = "market",
    reps: int = 500,
    window: Tuple[int, int] = (-1, 1),
    estimation_length: int = 120,
    estimation_gap: int = 10,
    seed: int = 42
) -> np.ndarray:
    """
    Generate empirical distribution of CARs for pseudo-events prior to the actual announcement.
    """
    pos = returns.index.searchsorted(pd.Timestamp(day0))
    min_pos = estimation_length + estimation_gap + abs(window[0]) + 2
    max_pos = pos - window[1] - 2

    if max_pos <= min_pos:
        return np.array([])

    rng = np.random.default_rng(seed)
    candidates = np.arange(min_pos, max_pos)
    sample_size = min(reps, len(candidates))
    chosen = rng.choice(candidates, size=sample_size, replace=False)

    placebo_cars = []
    for p in chosen:
        try:
            pseudo_date = returns.index[p]
            ar, _, _, _ = compute_abnormal_returns(
                returns, target=target, day0=pseudo_date, spec=spec,
                pre_window=abs(window[0]), post_window=window[1],
                estimation_length=estimation_length, estimation_gap=estimation_gap
            )
            w_slice = ar.loc[(ar.index >= window[0]) & (ar.index <= window[1])]
            if len(w_slice) == window[1] - window[0] + 1:
                placebo_cars.append(float(w_slice.sum()))
        except Exception:
            continue

    return np.asarray(placebo_cars)
