"""Interface en ligne de commande : python -m tradebot <commande> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from . import indicators as ind
from .config import load_config
from .utils import periods_per_year, timeframe_seconds


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _print_metrics(m: dict) -> None:
    pct = {"total_return", "max_drawdown", "var_95", "es_95", "vol_annual", "win_rate"}
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:<16} {v:+.2%}" if k in pct else f"  {k:<16} {v:.4f}")
        else:
            print(f"  {k:<16} {v}")


def _load_data(cfg, symbol: str, days: int | None):
    from .data.store import Store, fill_gaps, find_gaps

    store = Store(cfg.db_path)
    df = store.load(symbol, cfg.timeframe)
    if df.empty:
        print(f"Aucune donnée pour {symbol} {cfg.timeframe} dans {cfg.db_path} : lancez d'abord `fetch`.")
        sys.exit(1)
    if days:
        df = df.iloc[-int(days * 86400 / timeframe_seconds(cfg.timeframe)):]
    gaps = find_gaps(df, cfg.timeframe)
    if len(gaps) and cfg.asset_class == "crypto":
        print(f"  {len(gaps)} trou(s) détecté(s), {int(gaps['missing_bars'].sum())} barres reconstituées")
        df = fill_gaps(df, cfg.timeframe)
    return df


# --------------------------------------------------------------------------- commandes
def cmd_fetch(a, cfg):
    from .data.fetch import fetch_crypto_ohlcv
    from .data.store import Store

    store = Store(cfg.db_path)
    for symbol in (a.symbol or cfg.symbols):
        df = fetch_crypto_ohlcv(symbol, cfg.timeframe, days=a.days, exchange_id=cfg.exchange)
        n = store.upsert(symbol, cfg.timeframe, df)
        print(f"{symbol} {cfg.timeframe}: {n} barres enregistrées ({df.index.min()} -> {df.index.max()})")


def cmd_fetch_stock(a, cfg):
    from .data.fetch import fetch_stock_actions, fetch_stock_ohlcv
    from .data.store import Store

    store = Store(cfg.db_path)
    df = fetch_stock_ohlcv(a.ticker, interval=a.interval, period=f"{a.years}y")
    n = store.upsert(a.ticker, a.interval, df)
    print(f"{a.ticker} {a.interval}: {n} barres (ajustées splits/dividendes)")
    acts = fetch_stock_actions(a.ticker)
    print(f"  {int((acts.get('stock splits', pd.Series(dtype=float)) > 0).sum())} split(s), "
          f"{int((acts.get('dividends', pd.Series(dtype=float)) > 0).sum())} dividende(s) dans l'historique")


def cmd_backtest(a, cfg):
    from .backtest.engine import run_backtest
    from .strategies.base import get_strategy

    df = _load_data(cfg, a.symbol, a.days)
    strat = get_strategy(a.strategy or cfg.strategy, **cfg.strategy_params)
    prepared = strat.prepare(df)
    res = run_backtest(prepared, strat.signals(prepared), cfg.risk, periods_per_year(cfg.timeframe, cfg.asset_class))
    print(f"\nBacktest {a.symbol} {cfg.timeframe} {strat} sur {len(df)} barres "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    print("  (attention : in-sample, paramètres choisis à la main = optimiste ; voir `walkforward`)")
    _print_metrics(res.metrics)
    bh = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    print(f"  buy&hold        {bh:+.2%}")
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        res.equity.to_csv(Path(a.out) / "equity.csv")
        res.trades.to_csv(Path(a.out) / "trades.csv", index=False)
        print(f"  résultats écrits dans {a.out}/")


def cmd_walkforward(a, cfg):
    from .backtest.walkforward import walk_forward

    df = _load_data(cfg, a.symbol, a.days)
    wf = cfg.walkforward
    res = walk_forward(df, a.strategy or cfg.strategy, cfg.risk, periods_per_year(cfg.timeframe, cfg.asset_class),
                       train_bars=a.train or wf.train_bars, test_bars=a.test or wf.test_bars,
                       base_params=cfg.strategy_params, metric=wf.metric)
    print(f"\nWalk-forward {a.symbol} : {len(res.windows)} fenêtres, métrique={wf.metric}")
    pd.set_option("display.width", 200)
    print(res.windows[["test_start", "test_end", "best_params", f"is_{wf.metric}", f"oos_{wf.metric}",
                       "oos_return", "oos_trades"]].to_string(index=False))
    print("\nHors-échantillon concaténé (la seule estimation honnête) :")
    _print_metrics(res.metrics)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        res.oos_equity.to_csv(Path(a.out) / "wf_equity.csv")
        res.windows.to_csv(Path(a.out) / "wf_windows.csv", index=False)


def cmd_analyze(a, cfg):
    from .models.arima import arima_forecast
    from .models.stats import adf_test
    from .models.volatility import garch_forecast
    from .risk.metrics import expected_shortfall, var_historical

    df = _load_data(cfg, a.symbol, a.days)
    r = ind.log_returns(df["close"]).dropna()
    ppy = periods_per_year(cfg.timeframe, cfg.asset_class)
    print(f"\nAnalyse {a.symbol} {cfg.timeframe} ({len(df)} barres)")
    print(f"  vol réalisée annualisée (20 barres) : {float(ind.realized_vol(df['close'], 20, ppy).iloc[-1]):.2%}")
    print(f"  ATR courant : {float(ind.atr(df['high'], df['low'], df['close']).iloc[-1]):.2f} "
          f"({float(ind.atr(df['high'], df['low'], df['close']).iloc[-1] / df['close'].iloc[-1]):.2%} du prix)")
    print(f"  VaR 95 % par barre : {var_historical(r):.2%}   ES 95 % : {expected_shortfall(r):.2%}")
    adf_p = adf_test(df["close"])
    adf_r = adf_test(r)
    print(f"  ADF prix : p={adf_p['pvalue']:.3f} ({'stationnaire' if adf_p['stationary'] else 'non stationnaire, normal'})")
    print(f"  ADF rendements : p={adf_r['pvalue']:.3f} ({'stationnaire' if adf_r['stationary'] else 'non stationnaire'})")
    g = garch_forecast(r)
    print(f"  {g['model']} : vol prévue prochaine barre {g['vol']:.3%}"
          + (f", persistance alpha+beta={g['persistence']:.3f}" if "persistence" in g else ""))
    ar = arima_forecast(r.iloc[-1000:], max_p=2, max_q=2)
    print(f"  ARIMA{ar['order']} : rendement prévu {ar['forecast'][0]:+.4%} (direction {ar['direction']:+d}) "
          f"— faible pouvoir prédictif, à combiner")
    rsi_v = float(ind.rsi(df["close"]).iloc[-1])
    print(f"  RSI(14) : {rsi_v:.1f}")


def cmd_pairs(a, cfg):
    from .backtest.engine import run_backtest
    from .data.store import Store
    from .models.stats import engle_granger, johansen_test
    from .strategies.pairs import PairsStrategy

    store = Store(cfg.db_path)
    da, db = store.load(a.a, cfg.timeframe), store.load(a.b, cfg.timeframe)
    df = pd.DataFrame({"close_a": da["close"], "close_b": db["close"]}).dropna()
    if len(df) < 200:
        print("pas assez de données communes : lancez `fetch --symbol A --symbol B`")
        sys.exit(1)
    import numpy as np
    eg = engle_granger(np.log(df["close_a"]), np.log(df["close_b"]))
    jo = johansen_test(np.log(df))
    print(f"\nPaire {a.a} / {a.b} ({len(df)} barres)")
    print(f"  Engle-Granger : p={eg['pvalue']:.3f} hedge={eg['hedge_ratio']:.3f} "
          f"demi-vie={eg['half_life']:.1f} barres -> {'cointégrée' if eg['cointegrated'] else 'NON cointégrée'}")
    print(f"  Johansen : rang={jo['rank']} trace={jo['trace_stat'][0]:.2f} (crit 95 % {jo['crit_95'][0]:.2f})")
    if not eg["cointegrated"] and not jo["cointegrated"]:
        print("  -> pas de relation stable : le pairs trading n'a pas de sens ici.")
    strat = PairsStrategy(**cfg.strategy_params)
    prepared = strat.prepare(df)
    sig = strat.signals(prepared)
    # backtest simplifié : on trade le spread synthétique en unités du prix de A
    synth = pd.DataFrame(index=df.index)
    spread_px = df["close_a"] / df["close_b"] ** prepared["beta"].clip(lower=0.1)
    synth["close"] = spread_px
    synth["open"] = spread_px.shift(1).bfill()
    synth["high"] = synth[["open", "close"]].max(axis=1)
    synth["low"] = synth[["open", "close"]].min(axis=1)
    synth["volume"] = 0.0
    synth = synth.dropna()
    res = run_backtest(synth, sig.loc[synth.index], cfg.risk, periods_per_year(cfg.timeframe, cfg.asset_class),
                       use_stops=False)
    print(f"  backtest spread (approximation, nécessite vente à découvert, frais x2) :")
    _print_metrics(res.metrics)


def cmd_paper(a, cfg):
    from .data.fetch import make_exchange
    from .data.store import Store
    from .execution.broker import PaperBroker
    from .execution.live import TradingLoop
    from .strategies.base import get_strategy

    ex = make_exchange(cfg.exchange)  # accès public, sans clés
    broker = PaperBroker(cfg.risk.capital, cfg.risk.fee_rate, cfg.risk.slippage, price_source=ex,
                         timeframe=cfg.timeframe)
    loop = TradingLoop(cfg, broker, get_strategy(cfg.strategy, **cfg.strategy_params), store=Store(cfg.db_path))
    print(f"Paper trading {cfg.symbols} {cfg.timeframe} avec {cfg.risk.capital:.2f} (Ctrl+C pour arrêter)")
    loop.run(max_iterations=a.iterations)


def cmd_live(a, cfg):
    from .data.store import Store
    from .execution.broker import CcxtBroker
    from .execution.live import TradingLoop
    from .strategies.base import get_strategy

    if not cfg.sandbox and not a.i_understand_the_risks:
        print("Mode réel (sandbox: false) : relancez avec --i-understand-the-risks après avoir lu le README.")
        sys.exit(2)
    quote = cfg.symbols[0].split("/")[1]
    broker = CcxtBroker(cfg.exchange, cfg.api_key, cfg.api_secret, sandbox=cfg.sandbox, quote=quote,
                        max_order_notional=cfg.risk.capital * cfg.risk.max_position_pct * 1.05)
    loop = TradingLoop(cfg, broker, get_strategy(cfg.strategy, **cfg.strategy_params), store=Store(cfg.db_path))
    mode = "TESTNET" if cfg.sandbox else "RÉEL"
    print(f"Trading {mode} {cfg.symbols} {cfg.timeframe} — cash disponible : {broker.get_cash():.2f} {quote}")
    loop.run(max_iterations=a.iterations)


def cmd_status(a, cfg):
    p = Path(cfg.state_path)
    print(json.dumps(json.loads(p.read_text()), indent=2) if p.exists() else "aucun état enregistré")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradebot", description="Bot de trading quantitatif")
    p.add_argument("-c", "--config", default="config.yaml")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("fetch", help="télécharge l'OHLCV crypto (ccxt) dans SQLite")
    s.add_argument("--symbol", action="append")
    s.add_argument("--days", type=int, default=365)
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("fetch-stock", help="télécharge l'OHLCV d'une action (yfinance, ajusté)")
    s.add_argument("--ticker", required=True)
    s.add_argument("--interval", default="1d")
    s.add_argument("--years", type=int, default=5)
    s.set_defaults(fn=cmd_fetch_stock)

    for name, fn, helptxt in (("backtest", cmd_backtest, "backtest in-sample"),
                              ("walkforward", cmd_walkforward, "validation walk-forward"),
                              ("analyze", cmd_analyze, "stats : vol, ADF, GARCH, ARIMA, VaR")):
        s = sub.add_parser(name, help=helptxt)
        s.add_argument("--symbol", required=True)
        s.add_argument("--days", type=int)
        s.add_argument("--strategy")
        s.add_argument("--out")
        if name == "walkforward":
            s.add_argument("--train", type=int)
            s.add_argument("--test", type=int)
        s.set_defaults(fn=fn)

    s = sub.add_parser("pairs", help="test de cointégration + backtest pairs trading")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.set_defaults(fn=cmd_pairs)

    s = sub.add_parser("paper", help="paper trading en temps réel (prix publics, aucune clé)")
    s.add_argument("--iterations", type=int)
    s.set_defaults(fn=cmd_paper)

    s = sub.add_parser("live", help="trading réel/testnet via ccxt")
    s.add_argument("--iterations", type=int)
    s.add_argument("--i-understand-the-risks", action="store_true")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("status", help="affiche l'état persisté")
    s.set_defaults(fn=cmd_status)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    cfg = load_config(args.config)
    args.fn(args, cfg)


if __name__ == "__main__":
    main()
