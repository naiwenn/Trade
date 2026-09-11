"""Indicateurs techniques (tous causaux : la valeur en t n'utilise que t et le passé)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI de Wilder (lissage exponentiel alpha = 1/n)."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_down != 0, 100.0).where(avg_up.notna())


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower) / mid
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_width": width, "bb_pct": pct_b})


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> pd.DataFrame:
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    pk = 100 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    pd_ = pk.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"stoch_k": pk, "stoch_d": pd_})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def realized_vol(close: pd.Series, n: int = 20, periods_per_year: float | None = None) -> pd.Series:
    """Volatilité réalisée (écart-type glissant des log-rendements), annualisée si demandé."""
    v = log_returns(close).rolling(n, min_periods=n).std()
    return v * np.sqrt(periods_per_year) if periods_per_year else v


def add_all(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50, rsi_period: int = 14,
            bb_n: int = 20, atr_n: int = 14) -> pd.DataFrame:
    """Ajoute l'ensemble des indicateurs standards au DataFrame OHLCV."""
    out = df.copy()
    c, h, l = out["close"], out["high"], out["low"]
    out["ret"] = returns(c)
    out[f"ema_{ema_fast}"] = ema(c, ema_fast)
    out[f"ema_{ema_slow}"] = ema(c, ema_slow)
    out["sma_200"] = sma(c, 200)
    out = out.join(macd(c))
    out["rsi"] = rsi(c, rsi_period)
    out = out.join(bollinger(c, bb_n))
    out = out.join(stochastic(h, l, c))
    out["atr"] = atr(h, l, c, atr_n)
    out["atr_pct"] = out["atr"] / c
    out["rvol"] = realized_vol(c, 20)
    return out
