"""Tests statistiques : stationnarité (ADF), cointégration (Engle-Granger, Johansen),
demi-vie de retour à la moyenne, z-score glissant."""
from __future__ import annotations

import numpy as np
import pandas as pd


def adf_test(series: pd.Series, alpha: float = 0.05) -> dict:
    """Augmented Dickey-Fuller. H0 : racine unitaire (non stationnaire)."""
    from statsmodels.tsa.stattools import adfuller

    s = pd.Series(series).dropna()
    stat, pvalue, lags, nobs, crit, _ = adfuller(s, autolag="AIC")
    return {"stat": float(stat), "pvalue": float(pvalue), "lags": int(lags), "nobs": int(nobs),
            "crit": {k: float(v) for k, v in crit.items()}, "stationary": bool(pvalue < alpha)}


def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Régression y = a + b x par moindres carrés. Renvoie (b, a)."""
    x_ = np.asarray(x, dtype=float)
    y_ = np.asarray(y, dtype=float)
    b, a = np.polyfit(x_, y_, 1)
    return float(b), float(a)


def engle_granger(y: pd.Series, x: pd.Series, alpha: float = 0.05) -> dict:
    """Cointégration Engle-Granger : régression y ~ x puis ADF sur les résidus.

    Le test 'coint' de statsmodels utilise les bonnes valeurs critiques (MacKinnon)
    pour des résidus estimés, ce qu'un ADF naïf ne fait pas.
    """
    from statsmodels.tsa.stattools import coint

    df = pd.concat([y, x], axis=1).dropna()
    yy, xx = df.iloc[:, 0], df.iloc[:, 1]
    b, a = hedge_ratio_ols(yy, xx)
    spread = yy - b * xx - a
    stat, pvalue, crit = coint(yy, xx)
    return {"hedge_ratio": b, "intercept": a, "spread": spread, "stat": float(stat),
            "pvalue": float(pvalue), "crit": [float(c) for c in crit],
            "cointegrated": bool(pvalue < alpha), "half_life": half_life(spread)}


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """Test de Johansen (plusieurs séries). Renvoie le rang de cointégration à 95 %."""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    res = coint_johansen(prices.dropna().values, det_order, k_ar_diff)
    trace, crit95 = res.lr1, res.cvt[:, 1]
    rank = int(np.sum(trace > crit95))
    return {"trace_stat": trace.tolist(), "crit_95": crit95.tolist(), "rank": rank,
            "eigenvectors": res.evec, "cointegrated": rank > 0}


def half_life(spread: pd.Series) -> float:
    """Demi-vie d'un processus Ornstein-Uhlenbeck estimée par régression ds = k s_{t-1}."""
    s = pd.Series(spread).dropna()
    lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    if len(ds) < 3:
        return float("nan")
    k, _ = np.polyfit(lag.values, ds.values, 1)
    if k >= 0:
        return float("inf")
    return float(-np.log(2) / k)


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - m) / sd.replace(0.0, np.nan)
