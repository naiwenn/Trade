"""Volatilité conditionnelle : GARCH(1,1) (package arch) avec repli EWMA."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Volatilité EWMA façon RiskMetrics (lambda = 0.94 pour du quotidien)."""
    r = pd.Series(returns).dropna()
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def garch_forecast(returns: pd.Series, horizon: int = 1, dist: str = "t",
                   min_obs: int = 250) -> dict:
    """Ajuste un GARCH(1,1) sur des rendements (en %) et prévoit la vol des prochaines barres.

    Renvoie vol par barre (en fraction, pas en %). Repli EWMA si trop peu de données
    ou si l'optimisation échoue.
    """
    r = pd.Series(returns).dropna()
    if len(r) < min_obs:
        v = float(ewma_vol(r).iloc[-1]) if len(r) > 5 else float("nan")
        return {"model": "ewma", "vol": v, "params": {}}
    try:
        from arch import arch_model

        am = arch_model(r * 100, vol="GARCH", p=1, q=1, dist=dist, mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=horizon)
        var = fc.variance.iloc[-1].values
        return {"model": "garch11", "vol": float(np.sqrt(var[0]) / 100),
                "vol_path": (np.sqrt(var) / 100).tolist(),
                "params": {k: float(v) for k, v in res.params.items()},
                "persistence": float(res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0))}
    except Exception as exc:  # pragma: no cover - repli défensif
        return {"model": "ewma", "vol": float(ewma_vol(r).iloc[-1]), "params": {}, "error": str(exc)}
