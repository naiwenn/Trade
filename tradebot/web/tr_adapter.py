"""Adaptation des données Trade Republic aux modèles tradebot.

Particularités Trade Republic (France) :
- frais fixes : 1 EUR par ordre (achat ou vente), 0 EUR via plan d'épargne ;
- ordres classiques en actions entières (fractions uniquement via plans d'épargne) ;
- cotation via Lang & Schwarz (LSX), ~7h30-23h en semaine, spread plus large hors 9h-17h30.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

TR_FEE_PER_ORDER = 1.0
RANGE_TO_DAYS = {"1d": 1, "5d": 5, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "5y": 1825, "max": 3650}


def candles_to_df(history: dict) -> pd.DataFrame:
    """Réponse get_price_history -> DataFrame OHLCV indexé UTC (bougie en cours retirée)."""
    rows = history.get("candles", [])
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df["volume"] = df.get("volume", 0.0)
    df["volume"] = df["volume"].fillna(0.0)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")]
    res = history.get("resolution")
    if res and len(df):
        step = pd.Timedelta(milliseconds=int(res))
        if df.index[-1] + step > pd.Timestamp.now(tz="UTC"):
            df = df.iloc[:-1]  # bougie incomplète : jamais de décision dessus
    return df


def bars_per_year(history: dict, df: pd.DataFrame) -> float:
    """Estime le nombre de barres par an à partir de la résolution (actions : 252 j)."""
    res = history.get("resolution")
    if res:
        sec = int(res) / 1000
    elif len(df) > 2:
        sec = float((df.index[-1] - df.index[0]).total_seconds() / (len(df) - 1))
    else:
        return 252.0
    if sec >= 86400:
        return 252.0 * 86400 / sec
    return 252.0 * 8.5 * 3600 / sec  # ~8h30 de cotation LSX en heures ouvrées


def tr_position_size(capital: float, risk_per_trade: float, entry: float, stop: float,
                     fee_per_order: float = TR_FEE_PER_ORDER, fees_in_risk: bool = True,
                     max_position_pct: float = 0.5, whole_shares: bool = True,
                     min_notional: float = 1.0) -> dict:
    """Taille de position avec frais FIXES (Trade Republic) et actions entières.

    Renvoie {qty, notional, risk_eur, fee_round_trip, fee_pct_of_position, warnings}.
    """
    warnings: list[str] = []
    if entry <= 0 or capital <= 0:
        return {"qty": 0, "notional": 0.0, "risk_eur": 0.0, "fee_round_trip": 2 * fee_per_order,
                "fee_pct_of_position": None, "warnings": ["prix ou capital invalide"]}
    risk_amount = capital * risk_per_trade
    per_unit = abs(entry - stop)
    fee_rt = 2 * fee_per_order
    budget = risk_amount - fee_rt if fees_in_risk else risk_amount
    if fees_in_risk and budget <= 0:
        warnings.append(f"les frais aller-retour ({fee_rt:.2f} EUR) dépassent le budget de risque "
                        f"({risk_amount:.2f} EUR = {risk_per_trade:.0%} de {capital:.0f} EUR) : aucune position possible")
        return {"qty": 0, "notional": 0.0, "risk_eur": risk_amount, "fee_round_trip": fee_rt,
                "fee_pct_of_position": None, "warnings": warnings}
    qty = budget / per_unit if per_unit > 0 else 0.0
    max_qty = (capital * max_position_pct - fee_per_order) / entry
    qty = max(0.0, min(qty, max_qty))
    if whole_shares:
        qty = math.floor(qty)
    notional = qty * entry
    if qty <= 0 or notional < min_notional:
        warnings.append("position sous le minimum (une action entière / 1 EUR) : on ne trade pas")
        qty, notional = 0, 0.0
    fee_pct = fee_rt / notional if notional else None
    if fee_pct is not None and fee_pct > 0.01:
        warnings.append(f"frais aller-retour = {fee_pct:.1%} de la position : il faut > {fee_pct:.1%} de "
                        f"mouvement pour être à l'équilibre. Avec un petit capital, préférez les plans "
                        f"d'épargne (0 EUR de frais) ou peu de trades.")
    return {"qty": qty, "notional": round(notional, 2), "risk_eur": round(qty * per_unit + (fee_rt if fees_in_risk else 0), 2),
            "fee_round_trip": fee_rt, "fee_pct_of_position": fee_pct, "warnings": warnings}


def is_today_utc(ts: pd.Timestamp) -> bool:
    return ts.date() == datetime.now(timezone.utc).date()
