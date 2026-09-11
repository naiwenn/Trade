# Tradebot — bot de trading Trade Republic avec interface web

Bot de trading quantitatif en Python, branché sur le serveur MCP Trade Republic
[momolas/trade-republic-mcp-server](https://github.com/momolas/trade-republic-mcp-server)
(inclus en sous-module dans `vendor/`), avec une **interface navigateur** pour tout piloter :
authentification 2FA, portefeuille, analyse d'un instrument, backtest / walk-forward,
**auto-trading** (autopilote), ordres manuels et plans d'épargne.

> ⚠️ **Lisez ceci avant tout.**
> - Trade Republic n'a **pas d'API publique** : le serveur MCP utilise une API rétro-ingéniée
>   (projet pytr). Son usage peut contrevenir aux CGU ; votre compte peut être suspendu.
> - En mode **RÉEL**, le bot passe des ordres avec **votre argent**. Le mode par défaut est
>   **PAPIER** (simulation). Ne passez en réel qu'avec un montant que vous acceptez de perdre.
> - Aucune stratégie n'est garantie rentable. Ce projet n'est pas un conseil en investissement.

---

## 1. Ce que contient le projet

| Brique | Fichiers | Rôle |
|---|---|---|
| Serveur MCP Trade Republic | `vendor/trade-republic-mcp-server` (Node 22+) | 26 outils : portefeuille, prix, historique, carnet d'ordres, recherche, analyse technique, news/sentiment/fondamentaux, risque (Kelly, VaR), ordres, plans d'épargne, 2FA |
| Données | `tradebot/data/` | ccxt (crypto), yfinance (actions, ajusté splits/dividendes), FRED (macro), Fear & Greed (sentiment) ; stockage SQLite avec détection/remplissage des trous, index UTC |
| Indicateurs | `tradebot/indicators.py` | SMA, EMA, MACD, RSI (Wilder), Bollinger, stochastique, ATR, volatilité réalisée — tous causaux |
| Modèles | `tradebot/models/` | ADF, cointégration Engle-Granger & Johansen, demi-vie, GARCH(1,1) (repli EWMA), ARIMA, filtre de Kalman (hedge ratio dynamique), ACP + régression multi-facteurs, ML walk-forward (optionnel, scikit-learn) |
| Stratégies | `tradebot/strategies/` | `trend_combo` (vote EMA/MACD/RSI/Bollinger/stochastique + filtre de volatilité), `pairs` (z-score du spread cointégré) |
| Risque | `tradebot/risk/` | fraction fixe 1-2 %, Kelly fractionnaire plafonné, stop ATR, Sharpe/Sortino/Calmar, drawdown, VaR & Expected Shortfall, Markowitz |
| Backtest | `tradebot/backtest/` | signal au close → exécution à l'**open suivant** (zéro look-ahead), frais + slippage, stops intrabarre, taille recalculée sur le capital courant ; **walk-forward** rolling avec grille de paramètres et ratio OOS/IS |
| Exécution | `tradebot/execution/` | broker papier, broker ccxt (Binance testnet/réel), boucle live avec kill switch |
| Interface web | `tradebot/web/` | FastAPI + page unique (Chart.js) ; client MCP JSON-RPC ; adaptateur Trade Republic (frais fixes 1 €, actions entières) ; **autopilote** |
| Déploiement | `scripts/start.sh`, `Dockerfile`, `docker-compose.yml`, `deploy/tradebot.service` | VPS, Docker, systemd |

## 2. Installation (5 minutes)

```bash
git clone --recursive https://github.com/naiwenn/Trade.git && cd Trade
# (si déjà cloné sans --recursive : git submodule update --init)

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd vendor/trade-republic-mcp-server && npm ci && npm run build
cp .env.example .env         # puis renseigner TRADE_REPUBLIC_PHONE_NUMBER (+33…) et TRADE_REPUBLIC_PIN
cd ../..

./scripts/start.sh           # lance le serveur MCP (port 3006) puis l'interface (http://127.0.0.1:8080)
```

Au premier appel d'un outil authentifié, Trade Republic envoie un code (SMS ou app).
Dans l'onglet **Tableau de bord**, cliquez « Demander un code », saisissez-le, validez.
La session est ensuite rafraîchie automatiquement par le serveur MCP.

## 3. L'interface

- **Tableau de bord** : état MCP / authentification, équité, variation du jour vs kill switch,
  portefeuille (papier ou réel), journal de toutes les actions du bot.
- **Instrument & backtest** : recherche (nom, ticker, ISIN), graphique cours + EMA + Bollinger avec
  zones où la stratégie est en position, signal tradebot, analyse technique du serveur MCP, VaR/ES,
  **taille de position suggérée avec les frais**, news / sentiment / fondamentaux, backtest et
  walk-forward avec courbe d'équité vs buy & hold et liste des trades.
- **Autopilote & risque** : mode papier / réel (confirmation « REEL »), règles de risque
  (capital, % risqué par position, position max, perte journalière → kill switch, stop = k×ATR,
  stop suiveur, nombre max de positions, frais inclus ou non dans le risque), paramètres de la
  stratégie, intervalle des cycles, watchlist, **état de l'auto-trading par instrument**,
  bouton « Exécuter un cycle maintenant », arrêt d'urgence.
- **Ordres & plans d'épargne** : ordre manuel (marché / limite, confirmation), liste des ordres,
  création / annulation de plans d'épargne (DCA à 0 € de frais).

### Comment fonctionne l'auto-trading

À chaque cycle (intervalle réglable, 60 min par défaut), pour chaque instrument de la watchlist :

1. **Kill switch** : si l'équité du jour a perdu plus que `max_daily_loss_pct` (4 %), tout est
   liquidé, l'autopilote est désactivé et reste bloqué jusqu'à réarmement manuel.
2. Historique (bougies quotidiennes LSX) → indicateurs → **signal sur la dernière bougie close**
   (la bougie en cours est ignorée : pas de look-ahead).
3. **Stop** : si le dernier prix est sous le stop, vente. Sinon, si le stop suiveur est activé, le
   stop est remonté à `prix − k×ATR` (jamais abaissé). Les stops sont aussi vérifiés toutes les
   5 minutes entre deux cycles.
4. **Entrée** (signal long, pas de position, marché ouvert, nombre max de positions non atteint) :
   taille = budget de risque (`capital × risk_per_trade`) ÷ distance au stop, arrondie à l'**action
   entière**, plafonnée à `max_position_pct` du capital, refusée si nulle. Ordre au marché sur LSX.
5. **Sortie** sur signal neutre/baissier.
6. Une seule décision par bougie close ; tout est journalisé et persisté dans `data/web_state.json`
   (le bot survit à un redémarrage).

En mode papier, un portefeuille simulé (1 € de frais par ordre) remplace Trade Republic ; tout le
reste (données, signaux, stops, kill switch) est identique.

## 4. Ligne de commande (module crypto / recherche, indépendant de Trade Republic)

```bash
python -m tradebot fetch --symbol BTC/EUR --days 365      # OHLCV Binance -> SQLite
python -m tradebot fetch-stock --ticker AIR.PA             # action (yfinance, ajustée)
python -m tradebot analyze --symbol BTC/EUR                # vol, ADF, GARCH, ARIMA, VaR/ES
python -m tradebot backtest --symbol BTC/EUR               # in-sample (optimiste)
python -m tradebot walkforward --symbol BTC/EUR            # hors-échantillon (honnête)
python -m tradebot pairs --a BTC/EUR --b ETH/EUR           # cointégration + pairs trading
python -m tradebot paper                                   # paper trading Binance (prix publics)
python -m tradebot live                                    # testnet Binance (sandbox: true)
pytest -q                                                  # 35 tests, sans réseau
```

## 5. Où déployer (depuis la France)

Le bot travaille en bougies quotidiennes / cycles de plusieurs minutes : la latence ne compte pas.
Ce qui compte : une machine allumée 24/7, une IP stable, et vos identifiants bien protégés.

| Option | Coût | Avis |
|---|---|---|
| **VPS Hetzner (Falkenstein / Nuremberg, CX22)** | ≈ 4 €/mois | Le meilleur rapport qualité/prix en Europe, RGPD ok. Recommandé. |
| VPS OVH / Scaleway (France) | 3-7 €/mois | Données hébergées en France. |
| Oracle Cloud Free Tier (ARM, Francfort/Marseille) | 0 € | Gratuit mais création de compte parfois capricieuse. |
| Raspberry Pi / vieux PC à la maison | 0 € + électricité | Simple, mais dépend de votre box et de votre courant. |
| Votre ordinateur portable | 0 € | Seulement pour le mode papier : il doit rester allumé. |

Avec 100 €, un VPS à 4 €/mois représente déjà **4 % du capital par mois** : commencez par le mode
papier sur votre machine, et ne louez un serveur que si la stratégie tient en walk-forward.

### VPS en pratique (Debian/Ubuntu)

```bash
# sur le serveur
sudo apt update && sudo apt install -y python3-venv python3-pip git nodejs npm
sudo useradd -m -s /bin/bash tradebot && sudo -iu tradebot
git clone --recursive https://github.com/naiwenn/Trade.git /opt/tradebot && cd /opt/tradebot
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
(cd vendor/trade-republic-mcp-server && npm ci && npm run build && cp .env.example .env && nano .env)
exit
sudo cp /opt/tradebot/deploy/tradebot.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now tradebot
# depuis votre PC : tunnel SSH, puis http://localhost:8080
ssh -L 8080:127.0.0.1:8080 user@votre-vps
```

Ou avec Docker : `cp .env.example .env`, renseigner les identifiants, `docker compose up -d`.

**Sécurité** : l'interface écoute uniquement sur 127.0.0.1 (accès par tunnel SSH), n'exposez jamais
le port 8080 ni 3006 sur Internet ; `.env` n'est jamais committé ; activez le pare-feu (`ufw`) ;
mettez à jour le serveur.

## 6. La vérité sur 100 € chez Trade Republic

- **1 € par ordre** → un aller-retour coûte 2 €, soit **2 % du capital**. Chaque trade doit gagner
  plus de 2 % rien que pour être à l'équilibre. Le backtest et la taille de position en tiennent
  compte ; l'interface affiche l'avertissement à chaque suggestion.
- **Actions entières** pour les ordres classiques : avec un plafond de 50 € par position, seules les
  actions à moins de 50 € sont accessibles. Les fractions ne sont possibles que via plans d'épargne.
- **Plans d'épargne à 0 €** : mathématiquement, l'outil le plus favorable avec un petit capital
  (ETF monde en DCA). L'onglet dédié permet d'en créer.
- **Crypto** : Trade Republic propose BTC/ETH & co (1 € + spread) ; Binance (module ccxt, 0,1 %)
  est bien moins cher pour l'actif crypto — c'est le module `python -m tradebot`.
- Règle de risque 1-2 % : avec 100 €, c'est 1-2 € par trade… soit les frais. D'où l'option
  « frais inclus dans le risque » (désactivée par défaut, sinon le bot ne trade jamais) : c'est un
  choix conscient à faire, pas un bug.

## 7. Fiscalité (France, à vérifier sur impots.gouv.fr)

- Actions / ETF : plus-values au PFU 30 % (ou barème sur option), déclaration 2074 / 2042.
- Crypto : PFU 30 % sur les cessions contre euros (au-delà de 305 € de cessions par an), formulaire
  2086 ; comptes sur plateformes étrangères à déclarer (3916-bis) — Binance oui, Trade Republic
  (Allemagne) aussi pour le compte titres.

## 8. Feuille de route conseillée

1. Mode papier 4 à 8 semaines, watchlist de 2-3 instruments, journal lu chaque semaine.
2. Walk-forward sur chaque instrument : ratio Sharpe OOS/IS proche de 1 ou abandon.
3. Comparer au buy & hold et à un plan d'épargne : si le bot ne bat pas les deux nets de frais, DCA.
4. Réel avec 100 €, kill switch 4 %, un seul instrument, et acceptation de tout perdre.

## Licence

MIT pour ce dépôt. Le sous-module `vendor/trade-republic-mcp-server` est sous licence MIT
(© Alexander Rose).
