"""tradebot : bot de trading quantitatif modulaire (crypto + actions).

Modules :
- data       : récupération (ccxt, yfinance, FRED, Fear & Greed) et stockage SQLite
- indicators : SMA, EMA, MACD, RSI, Bollinger, stochastique, ATR, volatilité réalisée
- models     : ADF, cointégration (Engle-Granger, Johansen), GARCH, ARIMA, Kalman, ACP, ML
- strategies : combinaison d'indicateurs (vote), pairs trading
- risk       : dimensionnement (fraction fixe, Kelly), VaR/ES, Sharpe/Sortino, Markowitz
- backtest   : moteur sans look-ahead (exécution à l'open suivant), frais, slippage, walk-forward
- execution  : broker papier et broker ccxt (Binance), boucle live avec kill switch
"""

__version__ = "0.1.0"
