"""Sources de risque d'un portefeuille : ACP et régression multi-facteurs (type Fama-French)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pca_risk(returns: pd.DataFrame, n_components: int | None = None) -> dict:
    """ACP sur la matrice de covariance des rendements.

    explained : part de variance par composante ; loadings : poids de chaque actif
    sur chaque composante (la 1re est presque toujours le 'facteur marché').
    """
    r = returns.dropna()
    r = r - r.mean()
    cov = np.cov(r.values, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = n_components or len(vals)
    explained = vals / vals.sum()
    loadings = pd.DataFrame(vecs[:, :k], index=returns.columns, columns=[f"PC{i+1}" for i in range(k)])
    scores = pd.DataFrame(r.values @ vecs[:, :k], index=r.index, columns=loadings.columns)
    return {"explained": explained[:k].tolist(), "loadings": loadings, "scores": scores}


def factor_regression(asset_returns: pd.Series, factor_returns: pd.DataFrame, rf: float = 0.0) -> dict:
    """Régression r_asset - rf = alpha + sum(beta_i * facteur_i) + eps.

    factor_returns : DataFrame de facteurs (ex. marché, SMB, HML de Fama-French ou
    simplement BTC pour une altcoin). Renvoie alpha, betas, R2, t-stats.
    """
    import statsmodels.api as sm

    df = pd.concat([asset_returns - rf, factor_returns], axis=1).dropna()
    y = df.iloc[:, 0]
    X = sm.add_constant(df.iloc[:, 1:])
    res = sm.OLS(y, X).fit()
    return {"alpha": float(res.params.iloc[0]), "betas": res.params.iloc[1:].to_dict(),
            "tvalues": res.tvalues.to_dict(), "r2": float(res.rsquared), "resid_vol": float(res.resid.std())}
