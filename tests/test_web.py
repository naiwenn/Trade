"""Tests de l'interface web avec un serveur MCP simulé (aucun réseau)."""
import os
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import make_ohlcv
from tradebot.web import app as webapp
from tradebot.web.mcp_client import McpClient, McpError, TwoFactorRequired
from tradebot.web.tr_adapter import candles_to_df, tr_position_size

ISIN = "FR0000120073"


class FakeMcp:
    """Simule les outils du serveur MCP Trade Republic."""

    def __init__(self):
        df = make_ohlcv(n=400, regime=False, drift=0.003, vol=0.01, start="2024-01-01", freq="1D") * 0.05
        df["volume"] = 1000
        self.df = df
        self.price = float(df["close"].iloc[-1])
        self.need_2fa = False
        self.orders = []

    async def list_tools(self):
        return [f"tool{i}" for i in range(26)]

    async def call(self, tool, args=None):
        args = args or {}
        if self.need_2fa and tool != "enter_two_factor_code":
            raise TwoFactorRequired("TwoFactorCodeRequiredException: code sent to +33*****678")
        if tool == "enter_two_factor_code":
            self.need_2fa = False
            return {"message": "Authentication successful"}
        if tool == "get_price_history":
            rows = [{"time": int(t.value // 1_000_000), "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                     "volume": r.volume} for t, r in self.df.iterrows()]
            # bougie du jour (incomplète) ajoutée pour vérifier qu'elle est ignorée
            rows.append({"time": int(pd.Timestamp.now(tz="UTC").normalize().value // 1_000_000), "open": 1, "high": 1,
                         "low": 1, "close": 1, "volume": 0})
            return {"isin": args["isin"], "exchange": "LSX", "range": args["range"], "candles": rows, "resolution": 86400000}
        if tool == "get_price":
            return {"isin": args["isin"], "exchange": "LSX", "bid": self.price * 0.999, "ask": self.price * 1.001,
                    "last": self.price, "spread": 0.01, "spreadPercent": 0.2, "timestamp": "x"}
        if tool == "get_market_status":
            return {"isin": args["isin"], "exchange": "LSX", "status": "open", "isOpen": True, "hasBid": True, "hasAsk": True, "timestamp": "x"}
        if tool == "get_asset_info":
            return {"isin": args["isin"], "name": "Air Liquide SA", "shortName": "Air Liquide"}
        if tool == "search_assets":
            return {"results": [{"isin": ISIN, "name": "Air Liquide", "type": "stock"}], "totalCount": 1}
        if tool == "get_detailed_analysis":
            return {"isin": ISIN, "exchange": "LSX", "range": "3m", "currentPrice": self.price,
                    "summary": {"overallSignal": "buy", "confidence": 60, "score": 40, "bullishCount": 3, "bearishCount": 1, "neutralCount": 1},
                    "trend": {"direction": "uptrend", "strength": "moderate", "sma20": 1, "sma50": 1},
                    "signals": [{"indicator": "RSI", "signal": "hold", "strength": "weak", "reason": "neutral", "value": 55}],
                    "indicators": {}, "timestamp": "x"}
        if tool == "get_cash_balance":
            return {"availableCash": 100.0, "currency": "EUR"}
        if tool == "get_portfolio":
            return {"positions": [], "netValue": 0.0}
        if tool == "place_order":
            self.orders.append(args)
            return {"orderId": f"o{len(self.orders)}", "status": "placed", **args, "timestamp": "x"}
        if tool in ("get_news", "get_sentiment", "get_fundamentals"):
            return {"articles": [], "overallDirection": "neutral", "overallScore": 0, "confidence": "low"}
        if tool == "get_savings_plans":
            return {"savingsPlans": []}
        raise McpError(f"outil non simulé : {tool}")


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = FakeMcp()
    monkeypatch.setattr(webapp, "mcp", fake)
    monkeypatch.setattr(webapp, "state", webapp.State(tmp_path / "state.json"))
    webapp.history_cache.clear()
    webapp.names.clear()
    webapp.auth.update({"status": "unknown", "message": ""})
    with TestClient(webapp.app) as c:
        c.fake = fake
        yield c


def test_adapter_drops_forming_candle_and_sizing():
    fake = FakeMcp()
    import asyncio
    raw = asyncio.run(fake.call("get_price_history", {"isin": ISIN, "range": "1y"}))
    df = candles_to_df(raw)
    assert len(df) == 400 and df.index.tz is not None
    s = tr_position_size(100, 0.02, entry=20, stop=19, fees_in_risk=False, max_position_pct=0.5)
    assert s["qty"] == 2 and s["notional"] == 40 and s["fee_pct_of_position"] == pytest.approx(0.05)
    s2 = tr_position_size(100, 0.01, entry=20, stop=19, fees_in_risk=True)
    assert s2["qty"] == 0 and "frais aller-retour" in s2["warnings"][0]


def test_sse_parsing():
    txt = 'event: message\ndata: {"jsonrpc":"2.0","id":7,"result":{"ok":true}}\n\n'
    assert McpClient._parse_sse(txt, 7)["result"] == {"ok": True}


def test_dashboard_flow(client):
    st = client.get("/api/status").json()
    assert st["mcp_tools"] == 26 and st["dry_run"] is True and st["equity"] == 100.0

    assert client.get("/api/search?q=air").json()["results"][0]["isin"] == ISIN

    inst = client.get(f"/api/instrument/{ISIN}?range=1y").json()
    assert inst["bars"] == 400 and inst["current_signal"] == 1
    assert inst["suggestion"]["sizing"]["qty"] >= 1
    assert inst["analysis"]["summary"]["overallSignal"] == "buy"

    bt = client.post("/api/backtest", json={"isin": ISIN, "range": "1y", "walkforward": True, "train_bars": 150, "test_bars": 50}).json()
    assert bt["metrics"]["n_trades"] >= 1 and "walkforward" in bt and "metrics" in bt["walkforward"]

    # ---- auto-trading (papier)
    client.post("/api/watchlist", json={"isin": ISIN})
    rep = client.post("/api/autopilot/run").json()
    assert rep[ISIN]["status"].startswith("ACHAT"), rep
    pf = client.get("/api/portfolio").json()
    assert len(pf["positions"]) == 1 and pf["cash"] < 100
    assert webapp.state.data["stops"][ISIN] < client.fake.price

    rep2 = client.post("/api/autopilot/run").json()
    assert "déjà traitée" in rep2[ISIN]["status"]

    # trailing stop : le prix monte -> le stop remonte
    old_stop = webapp.state.data["stops"][ISIN]
    client.fake.price *= 1.10
    client.post("/api/autopilot/run")
    assert webapp.state.data["stops"][ISIN] > old_stop

    # stop touché entre deux cycles -> vente
    client.fake.price = old_stop * 0.9
    import asyncio
    hits = asyncio.run(webapp.check_stops())
    assert hits == [ISIN] and client.get("/api/portfolio").json()["positions"] == []

    # ordre manuel papier puis kill switch
    r = client.post("/api/orders", json={"isin": ISIN, "side": "buy", "qty": 1, "confirm": True})
    assert r.status_code == 200 and r.json()["status"] == "simulated"
    webapp.state.data["day_start_equity"] = 10_000.0
    webapp.state.data["day"] = webapp.datetime.now(webapp.timezone.utc).date().isoformat()
    rep3 = client.post("/api/autopilot/run").json()
    assert rep3 == {"halted": True}
    st = client.get("/api/status").json()
    assert st["halted"] is True and st["paper"]["positions"] == {}
    assert client.post("/api/autopilot/run").status_code == 400
    client.post("/api/settings", json={"halted": False})
    assert client.get("/api/status").json()["halted"] is False


def test_live_mode_sends_orders_and_2fa(client):
    client.post("/api/settings", json={"dry_run": False})
    client.post("/api/watchlist", json={"isin": ISIN})
    rep = client.post("/api/autopilot/run").json()
    assert rep[ISIN]["status"].startswith("ACHAT")
    assert client.fake.orders and client.fake.orders[0]["orderType"] == "buy" and client.fake.orders[0]["exchange"] == "LSX"
    # plafond par position respecté (capital 100, max 50 %)
    assert client.fake.orders[0]["size"] * client.fake.price <= 52.5

    client.fake.need_2fa = True
    r = client.post("/api/auth/probe")
    assert r.status_code == 401 and client.get("/api/status").json()["auth"]["status"] == "2fa_required"
    assert client.post("/api/auth/2fa", json={"code": "1234"}).status_code == 200
    assert client.get("/api/status").json()["auth"]["status"] == "ok"


def test_order_requires_confirm(client):
    assert client.post("/api/orders", json={"isin": ISIN, "side": "buy", "qty": 1}).status_code == 400
    assert client.post("/api/orders", json={"isin": ISIN, "side": "sell", "qty": 1, "confirm": True}).status_code == 400
