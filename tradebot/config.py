"""Chargement de la configuration (YAML + variables d'environnement)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RiskConfig:
    capital: float = 100.0
    risk_per_trade: float = 0.01
    max_position_pct: float = 0.5
    max_daily_loss_pct: float = 0.04
    atr_stop_mult: float = 2.0
    fee_rate: float = 0.001
    slippage: float = 0.0005
    min_notional: float = 10.0
    kelly_fraction: float = 0.5

    def validate(self) -> None:
        if not 0 < self.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade doit être dans ]0, 0.05] (1-2 % recommandé)")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct doit être dans ]0, 1]")
        if self.capital <= 0:
            raise ValueError("capital doit être > 0")
        if self.fee_rate < 0 or self.slippage < 0:
            raise ValueError("fee_rate / slippage doivent être >= 0")


@dataclass
class WalkForwardConfig:
    train_bars: int = 2000
    test_bars: int = 500
    metric: str = "sharpe"


@dataclass
class Config:
    exchange: str = "binance"
    asset_class: str = "crypto"
    symbols: list[str] = field(default_factory=lambda: ["BTC/EUR"])
    timeframe: str = "1h"
    db_path: str = "data/market.db"
    state_path: str = "data/state.json"
    sandbox: bool = True
    strategy: str = "trend_combo"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(f"{self.exchange.upper()}_API_KEY") or None

    @property
    def api_secret(self) -> str | None:
        return os.environ.get(f"{self.exchange.upper()}_API_SECRET") or None


def _build(cls, data: dict[str, Any]):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_dotenv(path: str | Path = ".env") -> None:
    """Chargeur .env minimal (pas de dépendance python-dotenv)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | Path = "config.yaml") -> Config:
    load_dotenv()
    p = Path(path)
    raw: dict[str, Any] = {}
    if p.exists():
        raw = yaml.safe_load(p.read_text()) or {}
    risk = _build(RiskConfig, raw.pop("risk", {}) or {})
    risk.validate()
    wf = _build(WalkForwardConfig, raw.pop("walkforward", {}) or {})
    cfg = _build(Config, raw)
    cfg.risk = risk
    cfg.walkforward = wf
    return cfg
