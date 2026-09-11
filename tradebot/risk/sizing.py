"""Dimensionnement des positions.

Règle n°1 avec un petit capital : ne jamais risquer plus de 1-2 % du capital par
position, frais compris. Règle n°2 : vérifier que la position dépasse le minimum
d'ordre de l'exchange, sinon ne pas trader.
"""
from __future__ import annotations


def atr_stop(entry: float, atr: float, mult: float = 2.0, side: int = 1) -> float:
    """Niveau de stop à `mult` ATR du prix d'entrée."""
    return entry - side * mult * atr


def breakeven_move(fee_rate: float, slippage: float = 0.0) -> float:
    """Mouvement minimal (fraction) pour couvrir un aller-retour frais + slippage."""
    return 2 * (fee_rate + slippage)


def fixed_fraction_size(capital: float, risk_per_trade: float, entry: float, stop: float,
                        fee_rate: float = 0.0, slippage: float = 0.0, min_notional: float = 0.0,
                        max_position_pct: float = 1.0) -> float:
    """Quantité telle que (perte au stop + frais aller-retour) = risk_per_trade * capital.

    Renvoie 0 si la position résultante est sous le minimum d'ordre.
    """
    if entry <= 0 or capital <= 0:
        return 0.0
    risk_amount = capital * risk_per_trade
    per_unit_loss = abs(entry - stop) + entry * breakeven_move(fee_rate, slippage)
    if per_unit_loss <= 0:
        return 0.0
    qty = risk_amount / per_unit_loss
    max_qty = capital * max_position_pct / (entry * (1 + fee_rate + slippage))
    qty = min(qty, max_qty)
    if qty * entry < min_notional:
        return 0.0
    return float(qty)


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Critère de Kelly : f* = p - (1-p)/b avec b = gain moyen / perte moyenne.

    avg_win et avg_loss en valeur absolue (fractions ou montants, même unité).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    b = avg_win / avg_loss
    f = win_rate - (1 - win_rate) / b
    return max(0.0, min(1.0, f))


def kelly_size(capital: float, win_rate: float, avg_win: float, avg_loss: float,
               fraction: float = 0.5, cap: float = 0.25) -> float:
    """Montant à engager selon un Kelly fractionnaire (demi-Kelly par défaut), plafonné.

    Kelly plein est beaucoup trop agressif : l'estimation de p et b est bruitée et
    la variance est énorme. Demi-Kelly conserve ~75 % de la croissance avec bien
    moins de drawdown.
    """
    f = kelly_fraction(win_rate, avg_win, avg_loss) * fraction
    return capital * min(f, cap)
