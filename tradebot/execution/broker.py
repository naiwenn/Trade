"""Brokers : papier (simulation) et ccxt (Binance & co.).

Sécurités du broker réel :
- clés API lues depuis l'environnement, jamais en dur ;
- mode sandbox (testnet) par défaut ;
- plafond de notional par ordre et vérification du minimum de l'exchange ;
- arrondi de la quantité à la précision du marché.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..data.fetch import OHLCV_COLS, make_exchange
from ..utils import timeframe_seconds

log = logging.getLogger("tradebot.broker")


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    order_id: str = ""


class Broker(ABC):
    @abstractmethod
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame: ...

    @abstractmethod
    def get_price(self, symbol: str) -> float: ...

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def get_position(self, symbol: str) -> float: ...

    @abstractmethod
    def market_order(self, symbol: str, side: str, qty: float) -> Fill | None: ...

    def equity(self, symbols: list[str]) -> float:
        return self.get_cash() + sum(self.get_position(s) * self.get_price(s) for s in symbols)


def _drop_forming_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Retire la bougie en cours (incomplète) : on ne décide que sur des bougies closes."""
    if len(df) and df.index[-1] + pd.Timedelta(seconds=timeframe_seconds(timeframe)) > pd.Timestamp.now(tz="UTC"):
        return df.iloc[:-1]
    return df


def _ohlcv_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", *OHLCV_COLS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").astype(float)


class PaperBroker(Broker):
    """Simule les exécutions au dernier prix connu, avec frais et slippage.

    price_source : objet avec fetch_ohlcv/fetch_ticker (ex. exchange ccxt public sans clés)
    ou None pour fonctionner avec des prix injectés via set_price (tests).
    """

    def __init__(self, cash: float, fee_rate: float = 0.001, slippage: float = 0.0005,
                 price_source=None, timeframe: str = "1h"):
        self.cash = cash
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.src = price_source
        self.timeframe = timeframe
        self.positions: dict[str, float] = {}
        self._prices: dict[str, float] = {}
        self.fills: list[Fill] = []

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def fetch_ohlcv(self, symbol, timeframe, limit=300):
        if self.src is None:
            raise RuntimeError("PaperBroker sans source de prix")
        df = _ohlcv_frame(self.src.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
        df = _drop_forming_candle(df, timeframe)
        self._prices[symbol] = float(df["close"].iloc[-1])
        return df

    def get_price(self, symbol):
        if self.src is not None:
            try:
                self._prices[symbol] = float(self.src.fetch_ticker(symbol)["last"])
            except Exception as exc:  # pragma: no cover
                log.warning("ticker indisponible pour %s : %s", symbol, exc)
        return self._prices[symbol]

    def get_cash(self):
        return self.cash

    def get_position(self, symbol):
        return self.positions.get(symbol, 0.0)

    def market_order(self, symbol, side, qty):
        px = self.get_price(symbol)
        exec_px = px * (1 + self.slippage) if side == "buy" else px * (1 - self.slippage)
        notional = qty * exec_px
        fee = notional * self.fee_rate
        if side == "buy":
            if notional + fee > self.cash + 1e-9:
                log.warning("cash insuffisant pour %s %s x %.6f", side, symbol, qty)
                return None
            self.cash -= notional + fee
            self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        else:
            have = self.positions.get(symbol, 0.0)
            qty = min(qty, have)
            if qty <= 0:
                return None
            self.cash += notional - fee
            self.positions[symbol] = have - qty
        fill = Fill(symbol, side, qty, exec_px, fee, f"paper-{len(self.fills)+1}")
        self.fills.append(fill)
        log.info("PAPER %s %s qty=%.6f px=%.2f fee=%.4f cash=%.2f", side.upper(), symbol, qty, exec_px, fee, self.cash)
        return fill


class CcxtBroker(Broker):
    def __init__(self, exchange_id: str, api_key: str, secret: str, sandbox: bool = True,
                 quote: str = "EUR", max_order_notional: float = 100.0):
        if not api_key or not secret:
            raise ValueError("clés API manquantes (variables BINANCE_API_KEY / BINANCE_API_SECRET)")
        self.ex = make_exchange(exchange_id, api_key, secret, sandbox=sandbox)
        self.ex.load_markets()
        self.quote = quote
        self.max_order_notional = max_order_notional
        self.sandbox = sandbox

    def fetch_ohlcv(self, symbol, timeframe, limit=300):
        df = _ohlcv_frame(self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
        return _drop_forming_candle(df, timeframe)

    def get_price(self, symbol):
        return float(self.ex.fetch_ticker(symbol)["last"])

    def get_cash(self):
        bal = self.ex.fetch_balance()
        return float(bal.get("free", {}).get(self.quote, 0.0))

    def get_position(self, symbol):
        base = symbol.split("/")[0]
        bal = self.ex.fetch_balance()
        return float(bal.get("free", {}).get(base, 0.0))

    def market_order(self, symbol, side, qty):
        market = self.ex.market(symbol)
        qty = float(self.ex.amount_to_precision(symbol, qty))
        px = self.get_price(symbol)
        notional = qty * px
        min_cost = (market.get("limits", {}).get("cost", {}) or {}).get("min") or 0.0
        min_amt = (market.get("limits", {}).get("amount", {}) or {}).get("min") or 0.0
        if qty < min_amt or notional < min_cost:
            log.warning("ordre sous le minimum de l'exchange (%s qty=%.8f notional=%.2f)", symbol, qty, notional)
            return None
        if notional > self.max_order_notional:
            log.error("ordre refusé : notional %.2f > plafond %.2f", notional, self.max_order_notional)
            return None
        order = self.ex.create_order(symbol, "market", side, qty)
        fee = 0.0
        for f in order.get("fees") or ([order["fee"]] if order.get("fee") else []):
            if f and f.get("cost"):
                fee += float(f["cost"])
        avg = float(order.get("average") or order.get("price") or px)
        filled = float(order.get("filled") or qty)
        log.info("LIVE%s %s %s qty=%.6f px=%.2f fee=%.4f id=%s", " [SANDBOX]" if self.sandbox else "",
                 side.upper(), symbol, filled, avg, fee, order.get("id"))
        return Fill(symbol, side, filled, avg, fee, str(order.get("id", "")))
