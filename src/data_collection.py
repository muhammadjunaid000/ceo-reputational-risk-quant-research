"""
Data Acquisition, Validation, and Quality Control Module
Supports Yahoo Finance downloads with fallback to local CSV cache.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yfinance as yf


def download_market_data(
    tickers: List[str],
    start: str = "2025-11-01",
    end: str = "2026-11-15",
    cache_dir: Union[str, Path] = "data/raw",
    force_refresh: bool = False
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Download daily OHLCV market data for given tickers with local caching.

    Parameters
    ----------
    tickers : list of str
        List of stock and benchmark tickers.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format (exclusive for yfinance).
    cache_dir : str or Path
        Directory for local cache files.
    force_refresh : bool
        If True, ignores existing cache and attempts fresh download.

    Returns
    -------
    frames : dict of str -> pd.DataFrame
        Dictionary containing 'close', 'volume', 'open', 'high', 'low' dataframes.
    source : str
        Data provenance indicator ('Yahoo Finance download' or 'Local CSV cache').
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    field_keys = ["close", "volume", "open", "high", "low"]
    yf_fields = ["Close", "Volume", "Open", "High", "Low"]
    files = {k: cache_path / f"{k}.csv" for k in field_keys}

    # Check if complete cache already exists and fresh download is not forced
    all_cached = all(p.exists() for p in files.values())
    if all_cached and not force_refresh:
        try:
            frames = {k: pd.read_csv(files[k], index_col=0, parse_dates=True) for k in field_keys}
            # Verify cached tickers cover requested tickers
            cols = set(frames["close"].columns)
            if set(tickers).issubset(cols):
                return frames, "Local CSV cache"
        except Exception:
            pass  # Fallback to network download if cache is unreadable

    # Attempt download via yfinance
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            progress=False,
            group_by="column",
            threads=True
        )
        if raw.empty:
            raise RuntimeError("Yahoo Finance returned an empty dataset.")

        frames = {}
        if isinstance(raw.columns, pd.MultiIndex):
            # Normal multi-ticker download
            for k_key, yf_key in zip(field_keys, yf_fields):
                if yf_key in raw.columns.levels[0]:
                    frame = raw[yf_key].copy()
                    frame.index = pd.to_datetime(frame.index).tz_localize(None)
                    frame.sort_index(inplace=True)
                    frame.to_csv(files[k_key])
                    frames[k_key] = frame
        else:
            # Single ticker download
            for k_key, yf_key in zip(field_keys, yf_fields):
                if yf_key in raw.columns:
                    frame = raw[[yf_key]].rename(columns={yf_key: tickers[0]}).copy()
                    frame.index = pd.to_datetime(frame.index).tz_localize(None)
                    frame.sort_index(inplace=True)
                    frame.to_csv(files[k_key])
                    frames[k_key] = frame

        # Also mirror into data/ if cache_dir is data/raw for backward compatibility
        alt_path = Path("data")
        if alt_path.exists():
            for k, f in frames.items():
                try:
                    f.to_csv(alt_path / f"{k}.csv")
                except Exception:
                    pass

        return frames, "Yahoo Finance download"

    except Exception as exc:
        if all_cached:
            frames = {k: pd.read_csv(files[k], index_col=0, parse_dates=True) for k in field_keys}
            return frames, "Local CSV cache (fallback after download error)"
        raise RuntimeError(
            f"Market data download failed ({exc}) and no local cache was found in {cache_path}."
        )


def validate_data_quality(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Perform rigorous data quality checks on the price matrix.

    Returns DataFrame of diagnostic metrics per asset.
    """
    quality = pd.DataFrame({
        "observations": prices.notna().sum(),
        "missing_pct": prices.isna().mean() * 100.0,
        "first_date": [prices[c].first_valid_index() for c in prices.columns],
        "latest_date": [prices[c].last_valid_index() for c in prices.columns],
        "nonpositive_prices": [(prices[c].dropna() <= 0).sum() for c in prices.columns],
        "zero_prices": [(prices[c].dropna() == 0).sum() for c in prices.columns]
    }, index=prices.columns)
    return quality


def load_governance_data(
    filepath: Union[str, Path] = "data/governance_inputs.csv"
) -> Optional[pd.DataFrame]:
    """
    Load external CEO governance and key-person risk metrics.
    Ensures transparent sourcing without placeholder imputation.
    """
    p = Path(filepath)
    if not p.exists():
        return None

    gov = pd.read_csv(p)
    required_cols = [
        "ticker", "ceo_key_person_dependence", "succession_readiness_gap",
        "board_independence_gap", "disclosure_uncertainty", "regulatory_exposure",
        "source_url", "assessment_date"
    ]
    missing = [c for c in required_cols if c not in gov.columns]
    if missing:
        raise ValueError(f"Governance dataset missing required columns: {missing}")

    # Validate score range [0, 100]
    score_cols = [
        "ceo_key_person_dependence", "succession_readiness_gap",
        "board_independence_gap", "disclosure_uncertainty", "regulatory_exposure"
    ]
    if not gov[score_cols].apply(lambda x: x.between(0, 100).all()).all():
        raise ValueError("All governance scores must strictly fall within 0-100 range.")

    return gov
