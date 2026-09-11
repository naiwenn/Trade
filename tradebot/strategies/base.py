"""Interface commune des stratégies.

Contrat : `signals(df)` renvoie une Series de positions cibles dans {-1, 0, +1}
calculée UNIQUEMENT avec l'information disponible à la clôture de chaque barre.
Le moteur de backtest décale ensuite d'une barre et exécute à l'open suivant :
c'est ce qui élimine le look-ahead bias.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

STRATEGIES: dict[str, type["Strategy"]] = {}


def register(cls):
    STRATEGIES[cls.name] = cls
    return cls


class Strategy(ABC):
    name: str = "base"
    default_params: dict[str, Any] = {}
    #: grille explorée par le walk-forward
    param_grid: dict[str, list[Any]] = {}

    def __init__(self, **params: Any):
        self.params = {**self.default_params, **params}

    @abstractmethod
    def signals(self, df: pd.DataFrame) -> pd.Series: ...

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes nécessaires (indicateurs) ; surchargeable."""
        return df

    def __repr__(self) -> str:
        return f"{self.name}({self.params})"


def get_strategy(name: str, **params: Any) -> Strategy:
    if name not in STRATEGIES:
        raise KeyError(f"stratégie inconnue : {name} (disponibles : {list(STRATEGIES)})")
    return STRATEGIES[name](**params)
