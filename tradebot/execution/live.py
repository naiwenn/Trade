"""Boucle de trading (papier ou réel).

À chaque clôture de bougie :
1. récupérer les N dernières bougies closes (la bougie en cours est ignorée) ;
2. calculer le signal de la stratégie sur la dernière bougie close ;
3. comparer à la position détenue ; si différent, dimensionner (fraction fixe,
   stop ATR, minimum d'ordre) et passer un ordre au marché ;
4. gérer le stop (vérifié à chaque passage sur le dernier prix) ;
5. kill switch : si la perte du jour dépasse max_daily_loss_pct, tout liquider et
   s'arrêter jusqu'à intervention humaine.
L'état (positions, stops, PnL du jour) est persisté en JSON pour survivre à un redémarrage.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import Config
from ..risk.sizing import atr_stop, fixed_fraction_size
from ..strategies.base import Strategy
from ..utils import timeframe_seconds
from .broker import Broker

log = logging.getLogger("tradebot.live")


class TradingLoop:
    def __init__(self, cfg: Config, broker: Broker, strategy: Strategy, store=None):
        self.cfg = cfg
        self.broker = broker
        self.strategy = strategy
        self.store = store
        self.state_path = Path(cfg.state_path)
        self.state = self._load_state()

    # ------------------------------------------------------------------ état
    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"positions": {}, "stops": {}, "day": None, "day_start_equity": None,
                "halted": False, "last_bar": {}}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2, default=str))

    # ------------------------------------------------------------------ kill switch
    def _check_kill_switch(self) -> bool:
        today = datetime.now(timezone.utc).date().isoformat()
        eq = self.broker.equity(self.cfg.symbols)
        if self.state.get("day") != today:
            self.state["day"], self.state["day_start_equity"] = today, eq
        start = self.state["day_start_equity"] or eq
        dd = eq / start - 1 if start else 0.0
        if dd < -self.cfg.risk.max_daily_loss_pct:
            log.error("KILL SWITCH : perte du jour %.2f%% > %.2f%%. Liquidation et arrêt.",
                      dd * 100, self.cfg.risk.max_daily_loss_pct * 100)
            for s in self.cfg.symbols:
                self._flatten(s, "kill_switch")
            self.state["halted"] = True
            self._save_state()
            return True
        return False

    # ------------------------------------------------------------------ ordres
    def _flatten(self, symbol: str, reason: str) -> None:
        qty = self.broker.get_position(symbol)
        if qty > 0:
            self.broker.market_order(symbol, "sell", qty)
            log.info("Sortie %s (%s)", symbol, reason)
        self.state["positions"].pop(symbol, None)
        self.state["stops"].pop(symbol, None)

    def _enter(self, symbol: str, price: float, atr_now: float) -> None:
        r = self.cfg.risk
        capital = self.broker.equity(self.cfg.symbols)
        stop = atr_stop(price, atr_now, r.atr_stop_mult, 1)
        qty = fixed_fraction_size(capital, r.risk_per_trade, price, stop, r.fee_rate, r.slippage,
                                  r.min_notional, r.max_position_pct)
        if qty <= 0:
            log.info("%s : position trop petite pour le minimum d'ordre, on passe.", symbol)
            return
        fill = self.broker.market_order(symbol, "buy", qty)
        if fill:
            self.state["positions"][symbol] = {"qty": fill.qty, "entry": fill.price, "ts": str(datetime.now(timezone.utc))}
            self.state["stops"][symbol] = atr_stop(fill.price, atr_now, r.atr_stop_mult, 1)

    # ------------------------------------------------------------------ une itération
    def step(self) -> dict:
        if self.state.get("halted"):
            log.warning("Bot arrêté par le kill switch : supprimer 'halted' dans %s pour relancer.", self.state_path)
            return {"halted": True}
        if self._check_kill_switch():
            return {"halted": True}
        report = {}
        for symbol in self.cfg.symbols:
            df = self.broker.fetch_ohlcv(symbol, self.cfg.timeframe, limit=400)
            if len(df) < 60:
                log.warning("%s : historique insuffisant (%d barres)", symbol, len(df))
                continue
            if self.store is not None:
                self.store.upsert(symbol, self.cfg.timeframe, df)
            last_ts = str(df.index[-1])
            prepared = self.strategy.prepare(df)
            sig = int(self.strategy.signals(prepared).iloc[-1])
            price = self.broker.get_price(symbol)
            atr_now = float(prepared["atr"].iloc[-1]) if "atr" in prepared else price * 0.02
            held = self.broker.get_position(symbol) > 0
            stop = self.state["stops"].get(symbol)

            # stop de protection (vérifié à chaque itération, pas seulement à la clôture)
            if held and stop is not None and price <= stop:
                self._flatten(symbol, f"stop {stop:.2f}")
                held = False

            # ne réagir qu'une seule fois par bougie close
            if self.state["last_bar"].get(symbol) != last_ts:
                self.state["last_bar"][symbol] = last_ts
                if sig > 0 and not held:
                    self._enter(symbol, price, atr_now)
                elif sig <= 0 and held:
                    self._flatten(symbol, "signal")
            report[symbol] = {"bar": last_ts, "signal": sig, "price": price, "held": self.broker.get_position(symbol) > 0,
                              "stop": self.state["stops"].get(symbol)}
        report["equity"] = self.broker.equity(self.cfg.symbols)
        self._save_state()
        log.info("step : %s", report)
        return report

    def run(self, poll_seconds: int | None = None, max_iterations: int | None = None) -> None:
        """Boucle infinie : un `step` juste après chaque clôture de bougie (+ vérif stops entre-temps)."""
        tf = timeframe_seconds(self.cfg.timeframe)
        poll = poll_seconds or max(30, tf // 12)
        it = 0
        log.info("Démarrage boucle %s %s (poll %ss)", self.cfg.symbols, self.cfg.timeframe, poll)
        while max_iterations is None or it < max_iterations:
            try:
                out = self.step()
                if out.get("halted"):
                    break
            except Exception:  # réseau, exchange... on log et on réessaie
                log.exception("erreur dans step()")
            it += 1
            if max_iterations is not None and it >= max_iterations:
                break
            time.sleep(poll)
