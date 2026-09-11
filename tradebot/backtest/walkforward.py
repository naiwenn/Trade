"""Validation walk-forward (anchored ou rolling).

Pour chaque fenêtre : on optimise les paramètres sur `train_bars` barres, puis on
les applique tels quels sur les `test_bars` barres suivantes, jamais vues. La
courbe hors-échantillon (OOS) concaténée est la seule estimation honnête de ce
que la stratégie aurait fait. Une cross-validation classique mélangerait passé et
futur et gonflerait artificiellement les résultats.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import RiskConfig
from ..risk.metrics import summary
from ..strategies.base import get_strategy
from .engine import run_backtest


@dataclass
class WalkForwardResult:
    oos_equity: pd.Series
    oos_returns: pd.Series
    windows: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:
        m = self.metrics
        return (f"WalkForwardResult(windows={len(self.windows)}, oos_return={m.get('total_return', 0):+.2%}, "
                f"oos_sharpe={m.get('sharpe', float('nan')):.2f}, mdd={m.get('max_drawdown', 0):.2%})")


def _score(result, metric: str) -> float:
    v = result.metrics.get(metric, float("nan"))
    if metric == "max_drawdown":
        v = -abs(v)
    return -np.inf if v is None or np.isnan(v) else v


def _grid(param_grid: dict) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def walk_forward(df: pd.DataFrame, strategy_name: str, risk: RiskConfig, periods_per_year: float,
                 train_bars: int = 2000, test_bars: int = 500, param_grid: dict | None = None,
                 base_params: dict | None = None, metric: str = "sharpe", warmup: int = 300,
                 anchored: bool = False) -> WalkForwardResult:
    base_params = base_params or {}
    strat_cls = type(get_strategy(strategy_name))
    grid = _grid(param_grid if param_grid is not None else strat_cls.param_grid)
    if len(df) < train_bars + test_bars:
        raise ValueError(f"pas assez de barres ({len(df)}) pour train={train_bars} + test={test_bars}")

    windows, oos_rets, oos_trades = [], [], []
    capital = risk.capital
    start = 0
    while start + train_bars + test_bars <= len(df):
        tr_start = 0 if anchored else start
        tr_end = start + train_bars
        te_end = tr_end + test_bars
        train = df.iloc[tr_start:tr_end]
        # meilleur jeu de paramètres en in-sample
        best, best_params, best_is = -np.inf, {}, None
        for params in grid:
            strat = get_strategy(strategy_name, **{**base_params, **params})
            prepared = strat.prepare(train)
            res = run_backtest(prepared, strat.signals(prepared), risk, periods_per_year)
            sc = _score(res, metric)
            if sc > best:
                best, best_params, best_is = sc, params, res
        # application hors-échantillon : on calcule les indicateurs sur un préfixe (warmup)
        # + la fenêtre test, mais on ne backteste QUE la fenêtre test.
        strat = get_strategy(strategy_name, **{**base_params, **best_params})
        ctx = df.iloc[max(0, tr_end - warmup):te_end]
        prepared = strat.prepare(ctx)
        sig = strat.signals(prepared)
        test_slice = prepared.iloc[-test_bars:]
        risk_now = RiskConfig(**{**risk.__dict__, "capital": capital})
        oos = run_backtest(test_slice, sig.loc[test_slice.index], risk_now, periods_per_year)
        capital = float(oos.equity.iloc[-1])
        windows.append({"train_start": train.index[0], "train_end": train.index[-1],
                        "test_start": test_slice.index[0], "test_end": test_slice.index[-1],
                        "best_params": best_params, f"is_{metric}": best,
                        f"oos_{metric}": oos.metrics.get(metric), "oos_return": oos.metrics["total_return"],
                        "oos_trades": oos.metrics.get("n_trades", 0)})
        oos_rets.append(oos.returns)
        oos_trades.append(oos.trades)
        start += test_bars

    rets = pd.concat(oos_rets)
    equity = risk.capital * (1 + rets).cumprod()
    trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    metrics = summary(equity, rets, periods_per_year, trades if len(trades) else None)
    win = pd.DataFrame(windows)
    if len(win):
        is_mean = np.nanmean(win[f"is_{metric}"].astype(float))
        oos_mean = np.nanmean(win[f"oos_{metric}"].astype(float))
        # ratio OOS/IS : proche de 1 = robuste ; << 1 = surapprentissage
        metrics["is_oos_ratio"] = float(oos_mean / is_mean) if is_mean and not np.isnan(is_mean) else float("nan")
    return WalkForwardResult(equity, rets, win, metrics, trades)
