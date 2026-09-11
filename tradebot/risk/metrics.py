"""Mesures de performance et de risque : Sharpe, Sortino, drawdown, VaR, Expected Shortfall."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def _clean(r) -> np.ndarray:
    return np.asarray(pd.Series(r).dropna(), dtype=float)


def sharpe(returns, periods_per_year: float, rf_annual: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - rf_annual / periods_per_year
    sd = excess.std(ddof=1)
    return float(np.sqrt(periods_per_year) * excess.mean() / sd) if sd > 0 else float("nan")


def sortino(returns, periods_per_year: float, rf_annual: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    excess = r - rf_annual / periods_per_year
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    return float(np.sqrt(periods_per_year) * excess.mean() / downside) if downside > 0 else float("nan")


def max_drawdown(equity) -> float:
    e = _clean(equity)
    if len(e) == 0:
        return float("nan")
    peak = np.maximum.accumulate(e)
    return float(np.min(e / peak - 1))


def calmar(returns, equity, periods_per_year: float) -> float:
    r = _clean(returns)
    e = _clean(equity)
    if len(r) == 0:
        return float("nan")
    years = len(r) / periods_per_year
    cagr = (e[-1] / e[0]) ** (1 / years) - 1 if years > 0 and e[0] > 0 else float("nan")
    mdd = max_drawdown(e)
    return float(cagr / abs(mdd)) if mdd < 0 else float("nan")


def var_historical(returns, alpha: float = 0.05) -> float:
    """VaR historique : perte (positive) dépassée avec probabilité alpha."""
    r = _clean(returns)
    return float(-np.quantile(r, alpha)) if len(r) else float("nan")


def var_parametric(returns, alpha: float = 0.05) -> float:
    """VaR gaussienne (moyenne + z_alpha * sigma)."""
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(-(r.mean() + sps.norm.ppf(alpha) * r.std(ddof=1)))


def expected_shortfall(returns, alpha: float = 0.05) -> float:
    """Expected Shortfall (CVaR) : perte moyenne dans les alpha pires cas."""
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    cutoff = np.quantile(r, alpha)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


def summary(equity: pd.Series, returns: pd.Series, periods_per_year: float,
            trades: pd.DataFrame | None = None) -> dict:
    e = _clean(equity)
    r = _clean(returns)
    out = {
        "total_return": float(e[-1] / e[0] - 1) if len(e) > 1 else float("nan"),
        "sharpe": sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": max_drawdown(e),
        "calmar": calmar(r, e, periods_per_year),
        "var_95": var_historical(r, 0.05),
        "es_95": expected_shortfall(r, 0.05),
        "vol_annual": float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan"),
        "n_bars": int(len(r)),
    }
    if trades is not None and len(trades):
        pnl = trades["pnl"]
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        out.update({
            "n_trades": int(len(trades)),
            "win_rate": float((pnl > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(-losses.mean()) if len(losses) else 0.0,
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
            "total_fees": float(trades["fees"].sum()),
            "expectancy": float(pnl.mean()),
        })
    else:
        out["n_trades"] = 0
    return out
