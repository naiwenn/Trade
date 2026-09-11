import numpy as np
import pandas as pd
import pytest

from tradebot.backtest.engine import run_backtest
from tradebot.backtest.walkforward import walk_forward
from tradebot.config import RiskConfig
from tradebot.strategies import PairsStrategy, TrendCombo, get_strategy

from conftest import make_ohlcv

PPY = 365 * 24


def test_no_look_ahead_signal_shift(ohlcv):
    """Un signal placé à la barre t doit être exécuté à l'open de t+1."""
    sig = pd.Series(0, index=ohlcv.index)
    sig.iloc[100] = 1
    sig.iloc[101:110] = 1
    risk = RiskConfig(capital=100, fee_rate=0.0, slippage=0.0, min_notional=0, risk_per_trade=0.05)
    res = run_backtest(ohlcv, sig, risk, PPY, use_stops=False)
    t = res.trades.iloc[0]
    assert t["entry_ts"] == ohlcv.index[101]
    assert t["entry"] == pytest.approx(ohlcv["open"].iloc[101])
    # dernier signal 1 au close de la barre 109, signal 0 au close de 110 -> sortie à l'open de 111
    assert t["exit_ts"] == ohlcv.index[111]
    assert t["exit"] == pytest.approx(ohlcv["open"].iloc[111])


def test_fees_reduce_pnl(ohlcv):
    strat = TrendCombo()
    prep = strat.prepare(ohlcv)
    sig = strat.signals(prep)
    base = dict(capital=100, min_notional=0, risk_per_trade=0.02)
    free = run_backtest(prep, sig, RiskConfig(fee_rate=0, slippage=0, **base), PPY)
    paid = run_backtest(prep, sig, RiskConfig(fee_rate=0.001, slippage=0.0005, **base), PPY)
    assert free.metrics["n_trades"] > 5
    assert paid.metrics["total_return"] < free.metrics["total_return"]
    assert paid.metrics["total_fees"] > 0


def test_risk_per_trade_bounded(ohlcv):
    strat = TrendCombo()
    prep = strat.prepare(ohlcv)
    risk = RiskConfig(capital=100, risk_per_trade=0.01, min_notional=0, fee_rate=0.001, slippage=0.0005)
    res = run_backtest(prep, strat.signals(prep), risk, PPY)
    # la perte d'un trade stoppé sans gap ne doit pas excéder ~1 % du capital en jeu (+ tolérance gap)
    stopped = res.trades[res.trades["reason"] == "stop"]
    assert len(stopped) > 0
    eq_before = [float(res.equity.loc[:ts].iloc[-2]) for ts in stopped["entry_ts"]]
    worst = (stopped["pnl"].values / np.array(eq_before)).min()
    assert worst > -0.03


def test_min_notional_blocks_tiny_trades(ohlcv):
    strat = TrendCombo()
    prep = strat.prepare(ohlcv)
    risk = RiskConfig(capital=100, risk_per_trade=0.01, min_notional=1e6)
    res = run_backtest(prep, strat.signals(prep), risk, PPY)
    assert res.metrics["n_trades"] == 0 and res.equity.iloc[-1] == 100


def test_equity_consistency(ohlcv):
    strat = TrendCombo()
    prep = strat.prepare(ohlcv)
    risk = RiskConfig(capital=100, min_notional=0)
    res = run_backtest(prep, strat.signals(prep), risk, PPY)
    assert res.equity.iloc[-1] == pytest.approx(100 + res.trades["pnl"].sum(), rel=1e-6)
    assert (res.equity > 0).all()


def test_short_allowed(ohlcv):
    strat = TrendCombo(allow_short=True)
    prep = strat.prepare(ohlcv)
    res = run_backtest(prep, strat.signals(prep), RiskConfig(capital=100, min_notional=0), PPY)
    assert (res.trades["side"] == -1).any()


def test_walk_forward_runs():
    df = make_ohlcv(n=2600, seed=11)
    res = walk_forward(df, "trend_combo", RiskConfig(capital=100, min_notional=0), PPY,
                       train_bars=1200, test_bars=400, param_grid={"ema_fast": [10, 20], "min_votes": [3]})
    assert len(res.windows) == 3
    assert len(res.oos_returns) == 1200
    assert "is_oos_ratio" in res.metrics
    # les fenêtres test sont disjointes et ordonnées
    assert (res.windows["test_start"].diff().dropna() > pd.Timedelta(0)).all()


def test_pairs_strategy_signals():
    rng = np.random.default_rng(9)
    x = np.cumsum(rng.normal(size=1500)) * 0.01
    b = 100 * np.exp(x)
    spread = np.zeros(1500)
    for i in range(1, 1500):
        spread[i] = 0.9 * spread[i - 1] + rng.normal(scale=0.01)
    a = 2 * b * np.exp(spread)
    df = pd.DataFrame({"close_a": a, "close_b": b})
    sig = PairsStrategy(window=100).signals(df)
    assert set(sig.unique()) <= {-1, 0, 1} and (sig != 0).sum() > 10


def test_registry():
    assert get_strategy("trend_combo").name == "trend_combo"
    with pytest.raises(KeyError):
        get_strategy("nope")
