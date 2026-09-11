import pandas as pd

from tradebot.config import Config, RiskConfig
from tradebot.execution.broker import PaperBroker
from tradebot.execution.live import TradingLoop
from tradebot.strategies import TrendCombo

from conftest import make_ohlcv


class FakeSource:
    """Source de prix : renvoie une fenêtre glissante de bougies synthétiques."""

    def __init__(self, df, cursor=500):
        self.df, self.cursor = df, cursor

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=300, since=None):
        win = self.df.iloc[max(0, self.cursor - limit):self.cursor]
        return [[int(ts.value // 1_000_000), r.open, r.high, r.low, r.close, r.volume] for ts, r in win.iterrows()]

    def fetch_ticker(self, symbol):
        return {"last": float(self.df["close"].iloc[self.cursor - 1])}


def test_paper_broker_fees():
    b = PaperBroker(100, fee_rate=0.001, slippage=0.0)
    b.set_price("BTC/EUR", 50.0)
    f = b.market_order("BTC/EUR", "buy", 1.0)
    assert f.fee == 0.05 and b.cash == 100 - 50.05 and b.get_position("BTC/EUR") == 1.0
    assert b.market_order("BTC/EUR", "buy", 10.0) is None  # cash insuffisant
    b.market_order("BTC/EUR", "sell", 1.0)
    assert b.get_position("BTC/EUR") == 0.0 and b.cash == 100 - 0.1


def test_trading_loop_runs_and_kill_switch(tmp_path):
    df = make_ohlcv(n=1500, seed=21, start="2020-01-01")
    src = FakeSource(df)
    cfg = Config(symbols=["X/EUR"], timeframe="1h", state_path=str(tmp_path / "state.json"),
                 risk=RiskConfig(capital=100, min_notional=0, max_daily_loss_pct=0.04))
    broker = PaperBroker(100, price_source=src, timeframe="1h")
    loop = TradingLoop(cfg, broker, TrendCombo(min_votes=3))
    entered = False
    for _ in range(600):
        out = loop.step()
        src.cursor += 1
        if broker.get_position("X/EUR") > 0:
            entered = True
        if out.get("halted"):
            break
    assert entered, "le bot doit avoir pris au moins une position"
    assert (tmp_path / "state.json").exists()
    # kill switch : on force une perte journalière > 4 %
    loop.state["halted"] = False
    loop.state["day_start_equity"] = broker.equity(["X/EUR"]) * 2
    assert loop.step()["halted"] is True
    assert broker.get_position("X/EUR") == 0.0
