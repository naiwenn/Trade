"""Petites fonctions utilitaires partagées."""
from __future__ import annotations

import pandas as pd

TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400, "1w": 604800,
}


def timeframe_seconds(tf: str) -> int:
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"timeframe inconnu : {tf}")
    return TIMEFRAME_SECONDS[tf]


def periods_per_year(timeframe: str, asset_class: str = "crypto") -> float:
    """Nombre de barres par an pour annualiser Sharpe/vol.

    crypto : marché 24/7 -> 365 jours ; actions : ~252 jours de bourse, 6.5 h/jour.
    """
    sec = timeframe_seconds(timeframe)
    if asset_class == "crypto":
        return 365 * 86400 / sec
    if sec >= 86400:
        return 252 * 86400 / sec
    return 252 * 6.5 * 3600 / sec


def to_utc_index(df: pd.DataFrame, col: str | None = None) -> pd.DataFrame:
    """Force un DatetimeIndex UTC (synchronisation des fuseaux horaires)."""
    out = df.copy()
    if col is not None:
        out[col] = pd.to_datetime(out[col], utc=True)
        out = out.set_index(col)
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx
    out.index.name = "ts"
    return out.sort_index()
