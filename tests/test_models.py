import numpy as np
import pandas as pd

from tradebot.models import (KalmanHedge, adf_test, arima_forecast, engle_granger, factor_regression,
                             garch_forecast, half_life, johansen_test, pca_risk)


def test_adf_random_walk_vs_noise():
    rng = np.random.default_rng(0)
    rw = pd.Series(np.cumsum(rng.normal(size=1000)))
    noise = pd.Series(rng.normal(size=1000))
    assert not adf_test(rw)["stationary"]
    assert adf_test(noise)["stationary"]


def test_cointegration_detected():
    rng = np.random.default_rng(3)
    x = pd.Series(np.cumsum(rng.normal(size=1500)) + 100)
    y = 2 * x + 5 + rng.normal(scale=1.0, size=1500)  # cointégré par construction
    eg = engle_granger(y, x)
    assert eg["cointegrated"] and abs(eg["hedge_ratio"] - 2) < 0.1
    assert 0 < eg["half_life"] < 20
    jo = johansen_test(pd.DataFrame({"y": y, "x": x}))
    assert jo["rank"] >= 1
    # deux marches aléatoires indépendantes : pas cointégrées (le plus souvent)
    z = pd.Series(np.cumsum(rng.normal(size=1500)))
    assert engle_granger(z, x)["pvalue"] > 0.01


def test_half_life_of_ou():
    rng = np.random.default_rng(4)
    s = np.zeros(3000)
    for i in range(1, 3000):
        s[i] = 0.9 * s[i - 1] + rng.normal()
    hl = half_life(pd.Series(s))
    assert 4 < hl < 10  # théorique : -ln2/ln(0.9) ~ 6.6


def test_kalman_tracks_beta():
    rng = np.random.default_rng(5)
    x = pd.Series(np.cumsum(rng.normal(size=2000)) + 50)
    y = 1.5 * x + 3 + rng.normal(scale=0.5, size=2000)
    out = KalmanHedge(delta=1e-4, obs_var=0.25).fit(x, y)
    assert abs(out["beta"].iloc[-1] - 1.5) < 0.1


def test_garch_and_ewma():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0, 0.01, 800))
    g = garch_forecast(r)
    assert g["model"] in ("garch11", "ewma") and 0.003 < g["vol"] < 0.03
    assert garch_forecast(r.iloc[:100])["model"] == "ewma"


def test_arima_runs():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.01, 300))
    out = arima_forecast(r, max_p=1, max_q=1)
    assert len(out["forecast"]) == 1 and out["direction"] in (-1, 0, 1)


def test_pca_and_factor_regression():
    rng = np.random.default_rng(8)
    mkt = rng.normal(0, 0.01, 1000)
    rets = pd.DataFrame({"A": mkt + rng.normal(0, 0.003, 1000), "B": 0.8 * mkt + rng.normal(0, 0.003, 1000),
                         "C": rng.normal(0, 0.01, 1000)})
    p = pca_risk(rets)
    assert p["explained"][0] > 0.4 and abs(sum(p["explained"]) - 1) < 1e-9
    fr = factor_regression(rets["B"], pd.DataFrame({"mkt": mkt}, index=rets.index))
    assert abs(fr["betas"]["mkt"] - 0.8) < 0.05 and fr["r2"] > 0.8
