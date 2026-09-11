"""Récupération de données : marché (crypto via ccxt, actions via yfinance),
fondamentales (yfinance), macro (FRED) et alternatives (Fear & Greed).

Toutes les fonctions renvoient des DataFrames indexés en UTC avec les colonnes
open, high, low, close, volume (pour l'OHLCV).
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from ..utils import timeframe_seconds, to_utc_index

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _empty_ohlcv() -> pd.DataFrame:
    df = pd.DataFrame(columns=OHLCV_COLS, dtype=float)
    df.index = pd.DatetimeIndex([], tz="UTC", name="ts")
    return df


# --------------------------------------------------------------------------- crypto
def make_exchange(exchange_id: str = "binance", api_key: str | None = None,
                  secret: str | None = None, sandbox: bool = False):
    import ccxt  # import paresseux : pas nécessaire pour les tests hors-ligne

    cls = getattr(ccxt, exchange_id)
    ex = cls({"apiKey": api_key, "secret": secret, "enableRateLimit": True,
              "options": {"defaultType": "spot"}})
    if sandbox:
        ex.set_sandbox_mode(True)
    return ex


def fetch_crypto_ohlcv(symbol: str, timeframe: str = "1h", since: datetime | None = None,
                       days: int | None = None, exchange=None, exchange_id: str = "binance",
                       max_bars: int = 200_000) -> pd.DataFrame:
    """Télécharge l'historique OHLCV avec pagination (ccxt limite à ~1000 barres/appel)."""
    ex = exchange or make_exchange(exchange_id)
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(days=days or 365)
    since_ms = int(since.timestamp() * 1000)
    step_ms = timeframe_seconds(timeframe) * 1000
    rows: list[list[float]] = []
    while len(rows) < max_bars:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last = batch[-1][0]
        if len(batch) < 2 or last + step_ms > time.time() * 1000:
            break
        since_ms = last + step_ms
        time.sleep(getattr(ex, "rateLimit", 100) / 1000)
    if not rows:
        return _empty_ohlcv()
    df = pd.DataFrame(rows, columns=["ts", *OHLCV_COLS])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    # La dernière bougie est peut-être encore en formation : on la retire pour ne
    # jamais utiliser une information incomplète (look-ahead bias).
    if len(df) and df.index[-1] + pd.Timedelta(seconds=timeframe_seconds(timeframe)) > pd.Timestamp.now(tz="UTC"):
        df = df.iloc[:-1]
    return df[OHLCV_COLS].astype(float)


# --------------------------------------------------------------------------- actions
def fetch_stock_ohlcv(ticker: str, interval: str = "1d", start: str | None = None,
                      end: str | None = None, period: str | None = None,
                      auto_adjust: bool = True) -> pd.DataFrame:
    """OHLCV actions via yfinance.

    auto_adjust=True ajuste les prix des splits et dividendes (indispensable pour
    ne pas voir de faux sauts de prix dans un backtest).
    """
    import yfinance as yf

    kw = {"interval": interval, "auto_adjust": auto_adjust, "progress": False}
    if period:
        kw["period"] = period
    else:
        kw["start"] = start or (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")
        if end:
            kw["end"] = end
    raw = yf.download(ticker, **kw)
    if raw is None or raw.empty:
        return _empty_ohlcv()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)[OHLCV_COLS]
    return to_utc_index(df).astype(float)


def fetch_stock_actions(ticker: str) -> pd.DataFrame:
    """Splits et dividendes (colonnes 'dividends', 'stock splits')."""
    import yfinance as yf

    actions = yf.Ticker(ticker).actions
    if actions is None or actions.empty:
        return pd.DataFrame(columns=["dividends", "stock splits"])
    return to_utc_index(actions.rename(columns=str.lower))


def fetch_fundamentals(ticker: str) -> dict:
    """Quelques fondamentaux (PER, capitalisation, marges...) via yfinance."""
    import yfinance as yf

    info = yf.Ticker(ticker).info or {}
    keys = ["marketCap", "trailingPE", "forwardPE", "priceToBook", "profitMargins",
            "returnOnEquity", "debtToEquity", "freeCashflow", "totalRevenue", "dividendYield", "beta"]
    return {k: info.get(k) for k in keys}


# --------------------------------------------------------------------------- macro
def fetch_fred_series(series_id: str) -> pd.Series:
    """Série FRED (ex. DFF = Fed funds, CPIAUCSL = CPI US, DGS10 = taux 10 ans).

    Utilise l'export CSV public : aucune clé nécessaire.
    """
    import urllib.request

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        csv = resp.read().decode()
    df = pd.read_csv(io.StringIO(csv), na_values=".")
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    s = df.set_index(date_col)[series_id].astype(float)
    s.index.name = "ts"
    return s


# --------------------------------------------------------------------------- alternatives
def fetch_fear_greed(limit: int = 0) -> pd.Series:
    """Crypto Fear & Greed Index (alternative.me) : proxy gratuit du sentiment."""
    import json
    import urllib.request

    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode())["data"]
    s = pd.Series({pd.to_datetime(int(d["timestamp"]), unit="s", utc=True): float(d["value"]) for d in data})
    s.index.name = "ts"
    return s.sort_index()
