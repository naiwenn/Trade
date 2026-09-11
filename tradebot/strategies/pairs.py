"""Pairs trading / arbitrage statistique sur deux actifs cointégrés.

Entrée : DataFrame avec colonnes close_a, close_b (alignées en UTC).
Le hedge ratio est estimé de manière causale (OLS glissant ou filtre de Kalman),
le spread est standardisé (z-score) et on trade le retour à la moyenne :
  z > +entry  -> short le spread (short A, long B)  => signal -1
  z < -entry  -> long le spread  (long A, short B)  => signal +1
  |z| < exit  -> sortie
Nécessite un compte permettant la vente à découvert (marge/futures) : sur le
spot Binance seul le côté long est exécutable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.kalman import KalmanHedge
from ..models.stats import rolling_zscore
from .base import Strategy, register


@register
class PairsStrategy(Strategy):
    name = "pairs"
    default_params = {"method": "kalman", "window": 100, "entry_z": 2.0, "exit_z": 0.5,
                      "stop_z": 4.0, "delta": 1e-4}
    param_grid = {"entry_z": [1.5, 2.0, 2.5], "exit_z": [0.25, 0.5, 1.0], "window": [60, 100, 200]}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()
        a, b = np.log(out["close_a"]), np.log(out["close_b"])
        if p["method"] == "kalman":
            kf = KalmanHedge(delta=p["delta"]).fit(b, a)
            out["beta"] = kf["beta"]
            out["spread"] = kf["spread"]
            out["z"] = rolling_zscore(out["spread"], p["window"])
        else:  # OLS glissant
            w = p["window"]
            cov = a.rolling(w).cov(b)
            var = b.rolling(w).var()
            out["beta"] = cov / var
            out["spread"] = a - out["beta"] * b
            out["z"] = rolling_zscore(out["spread"], w)
        return out

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        if "z" not in df.columns:
            df = self.prepare(df)
        z = df["z"].values
        pos = np.zeros(len(df), dtype=int)
        cur = 0
        for i in range(len(z)):
            if np.isnan(z[i]):
                continue
            if cur == 0:
                if z[i] > p["entry_z"] and z[i] < p["stop_z"]:
                    cur = -1
                elif z[i] < -p["entry_z"] and z[i] > -p["stop_z"]:
                    cur = 1
            elif abs(z[i]) < p["exit_z"] or abs(z[i]) > p["stop_z"]:
                cur = 0
            pos[i] = cur
        return pd.Series(pos, index=df.index, name="signal")
