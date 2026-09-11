"""Moteur de backtest barre par barre.

Garanties anti-biais :
- le signal calculé à la clôture de la barre t est exécuté à l'OPEN de t+1
  (jamais au close de t : ce serait du look-ahead) ;
- frais et slippage appliqués sur chaque exécution ;
- stop ATR vérifié sur le low/high intrabarre, exécuté au niveau du stop
  (ou à l'open s'il gappe au-delà) ;
- taille de position calculée avec le capital courant (fraction fixe) et
  refusée si elle est sous le minimum d'ordre ;
- pas de biais de survivance possible ici : on ne backteste que l'actif fourni
  (attention à ce biais si vous choisissez vous-même des actifs 'qui ont bien marché').
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RiskConfig
from ..risk.metrics import summary
from ..risk.sizing import atr_stop, fixed_fraction_size


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        m = self.metrics
        return (f"BacktestResult(return={m.get('total_return', float('nan')):+.2%}, "
                f"sharpe={m.get('sharpe', float('nan')):.2f}, mdd={m.get('max_drawdown', float('nan')):.2%}, "
                f"trades={m.get('n_trades', 0)})")


def run_backtest(df: pd.DataFrame, signals: pd.Series, risk: RiskConfig, periods_per_year: float,
                 atr_col: str = "atr", use_stops: bool = True) -> BacktestResult:
    """df : OHLCV (+ colonne atr) ; signals : position cible {-1,0,1} datée à la clôture."""
    df = df.copy()
    if atr_col not in df.columns:
        from ..indicators import atr as _atr
        df[atr_col] = _atr(df["high"], df["low"], df["close"], 14)
    # décalage d'une barre : décision au close t, exécution à l'open t+1
    target = signals.reindex(df.index).fillna(0).astype(int).shift(1).fillna(0).astype(int).values
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    atr_prev = df[atr_col].shift(1).values
    n = len(df)

    cash = risk.capital
    qty = 0.0            # signé (>0 long, <0 short)
    side = 0
    entry_px = 0.0
    entry_ts = None
    stop = np.nan
    fees_trade = 0.0
    equity = np.empty(n)
    positions = np.zeros(n, dtype=int)
    trades: list[dict] = []
    fee, slip = risk.fee_rate, risk.slippage
    cooldown_side = 0    # après un stop, on attend que le signal se réinitialise avant de ré-entrer

    def fill_price(px: float, buy: bool) -> float:
        return px * (1 + slip) if buy else px * (1 - slip)

    def close_position(px: float, ts, reason: str):
        nonlocal cash, qty, side, entry_px, entry_ts, stop, fees_trade
        exec_px = fill_price(px, buy=side < 0)
        notional = abs(qty) * exec_px
        f = notional * fee
        pnl_gross = (exec_px - entry_px) * qty
        cash += pnl_gross - f + (abs(qty) * entry_px)  # restitue le notional bloqué
        fees_trade += f
        trades.append({"entry_ts": entry_ts, "exit_ts": ts, "side": side, "qty": abs(qty),
                       "entry": entry_px, "exit": exec_px, "pnl": pnl_gross - fees_trade,
                       "fees": fees_trade, "return": (pnl_gross - fees_trade) / (abs(qty) * entry_px),
                       "reason": reason})
        qty, side, entry_px, entry_ts, stop, fees_trade = 0.0, 0, 0.0, None, np.nan, 0.0

    def open_position(px: float, ts, new_side: int, atr_now: float, capital_now: float):
        nonlocal cash, qty, side, entry_px, entry_ts, stop, fees_trade
        exec_px = fill_price(px, buy=new_side > 0)
        stop_lvl = atr_stop(exec_px, atr_now, risk.atr_stop_mult, new_side) if use_stops and not np.isnan(atr_now) \
            else exec_px * (1 - new_side * 0.05)
        q = fixed_fraction_size(capital_now, risk.risk_per_trade, exec_px, stop_lvl, fee, slip,
                                risk.min_notional, risk.max_position_pct)
        if q <= 0:
            return
        f = q * exec_px * fee
        cash -= q * exec_px + f     # bloque le notional (long) ; pour un short on bloque aussi le notional en garantie
        qty, side, entry_px, entry_ts, stop, fees_trade = q * new_side, new_side, exec_px, ts, stop_lvl, f

    for i in range(n):
        ts = df.index[i]
        # 1) stop intrabarre sur la position ouverte
        if side != 0 and use_stops and not np.isnan(stop):
            hit = (side > 0 and l[i] <= stop) or (side < 0 and h[i] >= stop)
            if hit:
                px = min(o[i], stop) if side > 0 else max(o[i], stop)
                cooldown_side = side
                close_position(px, ts, "stop")
        if cooldown_side != 0 and target[i] != cooldown_side:
            cooldown_side = 0
        # 2) changement de position cible -> exécution à l'open
        blocked = cooldown_side != 0 and target[i] == cooldown_side
        if i > 0 and target[i] != side and not blocked:
            if side != 0:
                close_position(o[i], ts, "signal")
            if target[i] != 0:
                equity_now = cash  # position fermée : cash = capital courant
                open_position(o[i], ts, int(target[i]), atr_prev[i], equity_now)
        # 3) valorisation mark-to-market au close
        mtm = cash + (abs(qty) * entry_px + (c[i] - entry_px) * qty if side != 0 else 0.0)
        equity[i] = mtm
        positions[i] = side

    if side != 0:
        close_position(c[-1], df.index[-1], "end")
        equity[-1] = cash
    eq = pd.Series(equity, index=df.index, name="equity")
    rets = eq.pct_change().fillna(0.0)
    tr = pd.DataFrame(trades, columns=["entry_ts", "exit_ts", "side", "qty", "entry", "exit", "pnl", "fees",
                                       "return", "reason"])
    metrics = summary(eq, rets, periods_per_year, tr)
    return BacktestResult(eq, rets, pd.Series(positions, index=df.index, name="position"), tr, metrics)
