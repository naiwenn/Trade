"""Stratégie de tendance par vote d'indicateurs.

Chaque indicateur (EMA, MACD, RSI, Bollinger, stochastique) donne un vote ; on
n'entre que si un nombre minimal de votes est réuni, et on filtre les régimes
de volatilité extrême. Pris isolément, ces indicateurs n'ont quasiment aucun
pouvoir prédictif : la combinaison + la gestion du risque font le travail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import Strategy, register


@register
class TrendCombo(Strategy):
    name = "trend_combo"
    default_params = {
        "ema_fast": 20, "ema_slow": 50, "rsi_period": 14, "rsi_max_entry": 70, "rsi_min_entry": 30,
        "bb_n": 20, "atr_n": 14, "min_votes": 3, "exit_votes": 1,
        "max_atr_pct": 0.08,      # pas d'entrée si ATR > 8 % du prix (marché chaotique)
        "allow_short": False,
    }
    param_grid = {"ema_fast": [10, 20, 30], "ema_slow": [50, 100], "min_votes": [3, 4],
                  "rsi_max_entry": [65, 75]}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()
        c, h, l = out["close"], out["high"], out["low"]
        out["ema_fast"] = ind.ema(c, p["ema_fast"])
        out["ema_slow"] = ind.ema(c, p["ema_slow"])
        out = out.join(ind.macd(c))
        out["rsi"] = ind.rsi(c, p["rsi_period"])
        out = out.join(ind.bollinger(c, p["bb_n"]))
        out = out.join(ind.stochastic(h, l, c))
        out["atr"] = ind.atr(h, l, c, p["atr_n"])
        out["atr_pct"] = out["atr"] / c
        return out

    def votes(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        p = self.params
        long_votes = (
            (df["ema_fast"] > df["ema_slow"]).astype(int)
            + (df["macd_hist"] > 0).astype(int)
            + ((df["rsi"] > 50) & (df["rsi"] < p["rsi_max_entry"])).astype(int)
            + (df["close"] > df["bb_mid"]).astype(int)
            + (df["stoch_k"] > df["stoch_d"]).astype(int)
        )
        short_votes = (
            (df["ema_fast"] < df["ema_slow"]).astype(int)
            + (df["macd_hist"] < 0).astype(int)
            + ((df["rsi"] < 50) & (df["rsi"] > p["rsi_min_entry"])).astype(int)
            + (df["close"] < df["bb_mid"]).astype(int)
            + (df["stoch_k"] < df["stoch_d"]).astype(int)
        )
        return long_votes, short_votes

    def signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        if "ema_fast" not in df.columns:
            df = self.prepare(df)
        lv, sv = self.votes(df)
        ready = df[["ema_slow", "rsi", "bb_mid", "stoch_d", "atr"]].notna().all(axis=1)
        calm = df["atr_pct"] <= p["max_atr_pct"]
        pos = np.zeros(len(df), dtype=int)
        cur = 0
        lv_, sv_ = lv.values, sv.values
        ready_, calm_ = ready.values, calm.values
        for i in range(len(df)):
            if not ready_[i]:
                pos[i] = 0
                continue
            if cur == 0:
                if lv_[i] >= p["min_votes"] and calm_[i]:
                    cur = 1
                elif p["allow_short"] and sv_[i] >= p["min_votes"] and calm_[i]:
                    cur = -1
            elif cur == 1 and lv_[i] <= p["exit_votes"]:
                cur = 0
            elif cur == -1 and sv_[i] <= p["exit_votes"]:
                cur = 0
            pos[i] = cur
        return pd.Series(pos, index=df.index, name="signal")
