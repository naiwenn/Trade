from .base import Strategy, get_strategy, register, STRATEGIES
from .trend_combo import TrendCombo
from .pairs import PairsStrategy

__all__ = ["Strategy", "get_strategy", "register", "STRATEGIES", "TrendCombo", "PairsStrategy"]
