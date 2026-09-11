import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n=3000, seed=0, drift=0.0002, vol=0.01, start="2024-01-01", freq="1h", regime=True):
    """Série synthétique : marche aléatoire log-normale avec régimes de tendance."""
    rng = np.random.default_rng(seed)
    if regime:
        # alternance de tendances haussières / baissières / plates pour donner du grain aux indicateurs
        mu = np.repeat(rng.choice([-0.0008, 0.0, 0.0012], size=n // 200 + 1), 200)[:n]
    else:
        mu = np.full(n, drift)
    r = rng.normal(mu, vol)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, vol / 4, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC", name="ts")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(10, 100, n)}, index=idx)


@pytest.fixture
def ohlcv():
    return make_ohlcv()
