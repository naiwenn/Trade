"""Optimisation de portefeuille façon Markowitz (long only, somme des poids = 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _solve(objective, n: int, bounds=None):
    w0 = np.full(n, 1 / n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = bounds or [(0.0, 1.0)] * n
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


def min_variance_weights(returns: pd.DataFrame, max_weight: float = 1.0) -> pd.Series:
    r = returns.dropna()
    cov = np.cov(r.values, rowvar=False)
    cov = cov / np.mean(np.diag(cov))  # normalisation : objectif O(1) pour l'optimiseur
    w = _solve(lambda w: w @ cov @ w, len(r.columns), [(0.0, max_weight)] * len(r.columns))
    return pd.Series(w, index=r.columns)


def max_sharpe_weights(returns: pd.DataFrame, periods_per_year: float, rf_annual: float = 0.0,
                       max_weight: float = 1.0) -> pd.Series:
    r = returns.dropna()
    mu = r.mean().values * periods_per_year - rf_annual
    cov = np.cov(r.values, rowvar=False) * periods_per_year

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu) / vol if vol > 0 else 0.0

    w = _solve(neg_sharpe, len(r.columns), [(0.0, max_weight)] * len(r.columns))
    return pd.Series(w, index=r.columns)
