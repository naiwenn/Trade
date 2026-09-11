import numpy as np
import pandas as pd
import pytest

from tradebot.risk import metrics as m
from tradebot.risk import portfolio as pf
from tradebot.risk import sizing as sz


def test_fixed_fraction_respects_risk():
    capital, risk = 100.0, 0.01
    entry, stop = 50.0, 48.0
    q = sz.fixed_fraction_size(capital, risk, entry, stop, fee_rate=0.001, slippage=0.0005, min_notional=0)
    loss_at_stop = q * (entry - stop) + q * entry * sz.breakeven_move(0.001, 0.0005)
    assert loss_at_stop == pytest.approx(capital * risk)


def test_fixed_fraction_min_notional_and_cap():
    # stop très large -> position minuscule -> sous le minimum -> 0
    assert sz.fixed_fraction_size(100, 0.01, 50, 25, min_notional=10) == 0.0
    # stop très serré -> plafonné à max_position_pct
    q = sz.fixed_fraction_size(100, 0.01, 50, 49.99, max_position_pct=0.5, min_notional=0)
    assert q * 50 <= 50.0 + 1e-9


def test_kelly():
    assert sz.kelly_fraction(0.5, 1, 1) == 0.0
    assert sz.kelly_fraction(0.6, 1, 1) == pytest.approx(0.2)
    assert sz.kelly_fraction(0.3, 1, 1) == 0.0
    assert sz.kelly_size(100, 0.6, 1, 1, fraction=0.5) == pytest.approx(10.0)
    assert sz.kelly_size(100, 0.9, 5, 1, fraction=1.0, cap=0.25) == 25.0


def test_metrics_sanity():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    eq = 100 * (1 + r).cumprod()
    s = m.summary(eq, r, 365 * 24)
    assert -1 < s["max_drawdown"] <= 0
    assert s["var_95"] > 0 and s["es_95"] >= s["var_95"]
    assert abs(m.var_parametric(r) - m.var_historical(r)) < 0.01
    assert m.sortino(r, 8760) >= m.sharpe(r, 8760)


def test_max_drawdown_exact():
    eq = pd.Series([100, 120, 90, 100, 130])
    assert m.max_drawdown(eq) == pytest.approx(-0.25)


def test_markowitz_weights_sum_to_one():
    rng = np.random.default_rng(2)
    rets = pd.DataFrame(rng.normal([0.001, 0.0005, 0.0], [0.02, 0.01, 0.005], (500, 3)), columns=list("ABC"))
    w = pf.min_variance_weights(rets)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()
    assert w["C"] > w["A"]  # l'actif le moins volatil pèse le plus
    w2 = pf.max_sharpe_weights(rets, 252)
    assert w2.sum() == pytest.approx(1.0)
