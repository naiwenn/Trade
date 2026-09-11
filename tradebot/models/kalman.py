"""Filtre de Kalman pour estimer dynamiquement le ratio de couverture d'une paire :
y_t = beta_t * x_t + alpha_t + bruit, avec (beta, alpha) suivant une marche aléatoire.

Classique en pairs trading : le hedge ratio s'adapte au lieu d'être figé par une
régression sur toute l'histoire (qui, elle, introduirait un look-ahead bias).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanHedge:
    def __init__(self, delta: float = 1e-4, obs_var: float = 1e-3):
        self.delta = delta          # variance de transition (réactivité du beta)
        self.obs_var = obs_var      # variance d'observation
        self.theta = np.zeros(2)    # [beta, alpha]
        self.P = np.eye(2)          # covariance de l'état
        self.Q = self.delta / (1 - self.delta) * np.eye(2)

    def update(self, x: float, y: float) -> tuple[float, float, float, float]:
        """Une étape prédiction/mise à jour. Renvoie (beta, alpha, résidu, sqrt(var_résidu))."""
        F = np.array([x, 1.0])
        # prédiction
        P_pred = self.P + self.Q
        y_hat = F @ self.theta
        e = y - y_hat
        S = F @ P_pred @ F + self.obs_var
        # mise à jour
        K = P_pred @ F / S
        self.theta = self.theta + K * e
        self.P = P_pred - np.outer(K, F) @ P_pred
        return float(self.theta[0]), float(self.theta[1]), float(e), float(np.sqrt(S))

    def fit(self, x: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Applique le filtre sur toute la série (causal : chaque ligne n'utilise que le passé)."""
        rows = [self.update(float(xi), float(yi)) for xi, yi in zip(x.values, y.values)]
        out = pd.DataFrame(rows, columns=["beta", "alpha", "spread", "spread_std"], index=x.index)
        out["z"] = out["spread"] / out["spread_std"]
        return out
