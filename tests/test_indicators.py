import numpy as np
import pandas as pd

from tradebot import indicators as ind


def test_sma_ema_basic():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    assert ind.sma(s, 3).iloc[-1] == 4.0
    assert np.isnan(ind.sma(s, 3).iloc[1])
    e = ind.ema(s, 3)
    assert e.iloc[-1] > 3.5 and np.isnan(e.iloc[1])


def test_rsi_bounds(ohlcv):
    r = ind.rsi(ohlcv["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    # série strictement croissante -> RSI 100
    up = pd.Series(np.arange(1, 50, dtype=float))
    assert ind.rsi(up).iloc[-1] == 100.0


def test_bollinger_contains_price(ohlcv):
    bb = ind.bollinger(ohlcv["close"]).dropna()
    assert (bb["bb_upper"] >= bb["bb_mid"]).all() and (bb["bb_lower"] <= bb["bb_mid"]).all()


def test_atr_positive(ohlcv):
    a = ind.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
    assert (a > 0).all()


def test_indicators_are_causal(ohlcv):
    """La valeur en t ne doit pas changer si on ajoute des données après t."""
    full = ind.add_all(ohlcv)
    part = ind.add_all(ohlcv.iloc[:2000])
    cols = [c for c in part.columns if c not in ohlcv.columns]
    pd.testing.assert_frame_equal(full.iloc[:2000][cols], part[cols])


def test_macd_stoch_shapes(ohlcv):
    assert list(ind.macd(ohlcv["close"]).columns) == ["macd", "macd_signal", "macd_hist"]
    st = ind.stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
    assert st["stoch_k"].between(0, 100).all()
