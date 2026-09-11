"""ARIMA pour la composante tendancielle (prévision à court terme des rendements)."""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd


def arima_forecast(series: pd.Series, steps: int = 1, order: tuple[int, int, int] | None = None,
                   max_p: int = 3, max_q: int = 3, d: int = 0) -> dict:
    """Prévision ARIMA. Si order est None, petite recherche (p, d, q) par AIC.

    Sur des rendements financiers, le gain est faible : à utiliser comme filtre de
    direction, jamais comme signal unique.
    """
    from statsmodels.tsa.arima.model import ARIMA

    s = pd.Series(series).dropna().astype(float)
    if order is None:
        best = (float("inf"), (1, d, 0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
                if p == q == 0:
                    continue
                try:
                    aic = ARIMA(s.values, order=(p, d, q)).fit().aic
                except Exception:
                    continue
                if aic < best[0]:
                    best = (aic, (p, d, q))
        order = best[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(s.values, order=order).fit()
        fc = res.get_forecast(steps=steps)
    mean = np.asarray(fc.predicted_mean)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    return {"order": order, "aic": float(res.aic), "forecast": mean.tolist(),
            "conf_int": ci.tolist(), "direction": int(np.sign(mean[0]))}
