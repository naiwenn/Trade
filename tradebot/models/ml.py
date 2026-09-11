"""Machine learning (optionnel, nécessite scikit-learn).

AVERTISSEMENT : les données financières sont très bruitées ; les modèles ML
surapprennent facilement. Ce module impose donc une validation walk-forward
(jamais de cross-validation aléatoire) et un 'embargo' entre train et test pour
éviter les fuites d'information liées au chevauchement des labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind


def build_features(df: pd.DataFrame, horizon: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Features causales + label = signe du rendement futur à `horizon` barres.

    Le label utilise le futur (shift négatif) : il ne doit servir QUE pour
    l'entraînement, jamais comme feature.
    """
    c = df["close"]
    X = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5, 10):
        X[f"ret_{k}"] = c.pct_change(k)
    X["rsi"] = ind.rsi(c, 14) / 100
    X["macd_hist"] = ind.macd(c)["macd_hist"] / c
    X["bb_pct"] = ind.bollinger(c)["bb_pct"]
    X["atr_pct"] = ind.atr(df["high"], df["low"], c) / c
    X["rvol"] = ind.realized_vol(c, 20)
    X["ema_ratio"] = ind.ema(c, 20) / ind.ema(c, 50) - 1
    y = np.sign(c.shift(-horizon) / c - 1)
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid].astype(int)


def walk_forward_classifier(X: pd.DataFrame, y: pd.Series, train_size: int = 1000,
                            test_size: int = 250, embargo: int = 5, model=None) -> pd.DataFrame:
    """Prédictions hors-échantillon par fenêtres glissantes.

    Renvoie un DataFrame (index test) avec y_true, y_pred, proba_up.
    """
    if model is None:
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install scikit-learn pour utiliser tradebot.models.ml") from exc
        model = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=50,
                                       random_state=0, n_jobs=-1)
    preds = []
    start = 0
    n = len(X)
    while start + train_size + embargo + test_size <= n:
        tr = slice(start, start + train_size)
        te = slice(start + train_size + embargo, start + train_size + embargo + test_size)
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])
        up_idx = list(model.classes_).index(1) if 1 in model.classes_ else None
        p_up = proba[:, up_idx] if up_idx is not None else np.zeros(len(proba))
        preds.append(pd.DataFrame({"y_true": y.iloc[te].values, "y_pred": np.where(p_up > 0.5, 1, -1),
                                   "proba_up": p_up}, index=X.index[te]))
        start += test_size
    if not preds:
        raise ValueError("Pas assez de données pour une fenêtre walk-forward")
    return pd.concat(preds)
