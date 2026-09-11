"""Stockage des séries temporelles dans SQLite.

Pour 100 EUR et quelques symboles en 1h, SQLite suffit largement. Le schéma est
identique à ce que l'on mettrait dans TimescaleDB/InfluxDB si le besoin grandit
(une ligne = (symbol, timeframe, ts, o, h, l, c, v), clé unique sur les 3 premiers).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ..utils import timeframe_seconds

SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    ts INTEGER NOT NULL,           -- epoch ms UTC
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, timeframe, ts)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_ts ON ohlcv(symbol, timeframe, ts);
"""


class Store:
    def __init__(self, path: str | Path = "data/market.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.executescript(SCHEMA)

    def upsert(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        rows = [(symbol, timeframe, int(ts.value // 1_000_000), float(r.open), float(r.high),
                 float(r.low), float(r.close), float(r.volume))
                for ts, r in df.iterrows()]
        self.conn.executemany(
            "INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def load(self, symbol: str, timeframe: str, start=None, end=None) -> pd.DataFrame:
        q = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=?"
        params: list = [symbol, timeframe]
        if start is not None:
            q += " AND ts >= ?"
            params.append(int(pd.Timestamp(start, tz="UTC").value // 1_000_000))
        if end is not None:
            q += " AND ts <= ?"
            params.append(int(pd.Timestamp(end, tz="UTC").value // 1_000_000))
        q += " ORDER BY ts"
        df = pd.read_sql_query(q, self.conn, params=params)
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df.set_index("ts")

    def symbols(self) -> list[tuple[str, str, int]]:
        cur = self.conn.execute(
            "SELECT symbol, timeframe, COUNT(*) FROM ohlcv GROUP BY symbol, timeframe")
        return [tuple(r) for r in cur.fetchall()]

    def last_ts(self, symbol: str, timeframe: str) -> pd.Timestamp | None:
        cur = self.conn.execute(
            "SELECT MAX(ts) FROM ohlcv WHERE symbol=? AND timeframe=?", (symbol, timeframe))
        v = cur.fetchone()[0]
        return None if v is None else pd.to_datetime(v, unit="ms", utc=True)

    def close(self) -> None:
        self.conn.close()


def find_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Liste les trous (barres manquantes) : colonnes start, end, missing_bars."""
    if len(df) < 2:
        return pd.DataFrame(columns=["start", "end", "missing_bars"])
    step = pd.Timedelta(seconds=timeframe_seconds(timeframe))
    idx = df.index
    delta = pd.Series(idx[1:] - idx[:-1], index=idx[1:])
    bad = delta[delta > step]
    return pd.DataFrame({
        "start": [t - d for t, d in zip(bad.index, bad.values)],
        "end": bad.index,
        "missing_bars": [int(d / step) - 1 for d in bad.values],
    })


def fill_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Reconstitue une grille régulière : close propagé, volume 0, OHLC = close précédent.

    À utiliser pour la crypto (24/7). Pour les actions, les week-ends ne sont PAS
    des trous : ne pas appliquer.
    """
    if df.empty:
        return df
    step = pd.Timedelta(seconds=timeframe_seconds(timeframe))
    full = pd.date_range(df.index[0], df.index[-1], freq=step, tz="UTC", name="ts")
    out = df.reindex(full)
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    return out
