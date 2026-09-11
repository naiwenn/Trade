"""Application web : tableau de bord + autopilote branchés sur le serveur MCP Trade Republic.

Lancer :  uvicorn tradebot.web.app:app --host 127.0.0.1 --port 8080
(ou  python -m tradebot.web)

Sécurité par défaut :
- dry_run = true : les ordres sont simulés dans un portefeuille papier, rien n'est envoyé ;
- kill switch : perte journalière > max_daily_loss_pct -> liquidation + arrêt ;
- tout ordre réel exige confirm=true et respecte les plafonds de taille.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import indicators as ind
from ..backtest.engine import run_backtest
from ..backtest.walkforward import walk_forward
from ..config import RiskConfig
from ..risk.metrics import expected_shortfall, var_historical
from ..risk.sizing import atr_stop
from ..strategies.base import get_strategy
from .mcp_client import McpClient, McpError, TwoFactorRequired
from .tr_adapter import TR_FEE_PER_ORDER, bars_per_year, candles_to_df, tr_position_size

log = logging.getLogger("tradebot.web")
STATIC = Path(__file__).parent / "static"
STATE_PATH = Path(os.environ.get("TRADEBOT_WEB_STATE", "data/web_state.json"))
MCP_URL = os.environ.get("MCP_URL", "http://localhost:3006/mcp")

DEFAULT_STATE: dict[str, Any] = {
    "dry_run": True,
    "watchlist": [],
    "autopilot": {"enabled": False, "interval_min": 60, "range": "1y", "strategy": "trend_combo",
                  "params": {"ema_fast": 20, "ema_slow": 50, "min_votes": 3, "rsi_max_entry": 70}},
    "risk": {"capital": 100.0, "risk_per_trade": 0.02, "max_position_pct": 0.5, "max_daily_loss_pct": 0.04,
             "atr_stop_mult": 2.0, "fees_in_risk": False, "trailing_stop": True, "max_positions": 3},
    "paper": {"cash": 100.0, "positions": {}},
    "stops": {},
    "day": None, "day_start_equity": None, "halted": False,
    "last_cycle": None,
    "last_stop_check": None,
    "instruments": {},
    "journal": [],
}


# --------------------------------------------------------------------------- utilitaires
def clean(o: Any) -> Any:
    """Rend un objet sérialisable en JSON (numpy, NaN, Timestamp)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else float(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, pd.Series):
        return clean(o.tolist())
    return o


class State:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(json.dumps(DEFAULT_STATE))
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                for k, v in loaded.items():
                    if isinstance(v, dict) and isinstance(self.data.get(k), dict):
                        self.data[k].update(v)
                    else:
                        self.data[k] = v
            except Exception:
                log.exception("état illisible, valeurs par défaut")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(clean(self.data), indent=2, ensure_ascii=False))

    def journal(self, kind: str, msg: str, **extra: Any) -> None:
        entry = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, "msg": msg, **clean(extra)}
        self.data["journal"] = ([entry] + self.data["journal"])[:300]
        log.info("[%s] %s %s", kind, msg, extra or "")
        self.save()


state = State(STATE_PATH)
mcp = McpClient(MCP_URL)
auth = {"status": "unknown", "message": ""}
names: dict[str, str] = {}
history_cache: dict[tuple[str, str], tuple[float, dict]] = {}


async def tool(name: str, args: dict | None = None) -> Any:
    try:
        res = await mcp.call(name, args)
        if name not in ("get_sentiment", "get_news", "get_fundamentals", "calculate_position_size", "get_risk_metrics"):
            auth["status"], auth["message"] = "ok", ""
        return res
    except TwoFactorRequired as exc:
        auth["status"], auth["message"] = "2fa_required", str(exc)
        raise HTTPException(401, str(exc))
    except McpError as exc:
        msg = str(exc)
        if "injoignable" in msg:
            auth["status"] = "mcp_down"
        auth["message"] = msg
        raise HTTPException(502, msg)


async def asset_name(isin: str) -> str:
    if isin not in names:
        try:
            info = await mcp.call("get_asset_info", {"isin": isin})
            names[isin] = info.get("shortName") or info.get("name") or isin
        except Exception:
            names[isin] = isin
    return names[isin]


async def history(isin: str, rng: str, max_age: float = 300) -> tuple[pd.DataFrame, dict]:
    key = (isin, rng)
    now = datetime.now().timestamp()
    if key in history_cache and now - history_cache[key][0] < max_age:
        raw = history_cache[key][1]
    else:
        raw = await tool("get_price_history", {"isin": isin, "range": rng, "exchange": "LSX"})
        history_cache[key] = (now, raw)
    return candles_to_df(raw), raw


async def last_price(isin: str) -> float:
    p = await tool("get_price", {"isin": isin, "exchange": "LSX"})
    return float(p.get("last") or p.get("ask") or p.get("bid"))


def risk_cfg() -> RiskConfig:
    r = state.data["risk"]
    return RiskConfig(capital=float(r["capital"]), risk_per_trade=float(r["risk_per_trade"]),
                      max_position_pct=float(r["max_position_pct"]), max_daily_loss_pct=float(r["max_daily_loss_pct"]),
                      atr_stop_mult=float(r["atr_stop_mult"]), fee_rate=0.0, slippage=0.001, min_notional=1.0)


def analyse_frame(df: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.Series, Any]:
    strat = get_strategy(state.data["autopilot"]["strategy"], **params)
    prepared = strat.prepare(df)
    return prepared, strat.signals(prepared), strat


# --------------------------------------------------------------------------- portefeuille (papier ou réel)
async def positions() -> dict[str, dict]:
    if state.data["dry_run"]:
        return state.data["paper"]["positions"]
    pf = await tool("get_portfolio", {})
    out = {}
    for p in pf.get("positions", []):
        out[p["instrumentId"]] = {"qty": float(p["netSize"]), "entry": float(p.get("averageCost") or 0),
                                  "value": float(p.get("netValue") or 0)}
    return out


async def equity() -> float:
    if state.data["dry_run"]:
        total = float(state.data["paper"]["cash"])
        for isin, pos in state.data["paper"]["positions"].items():
            try:
                total += pos["qty"] * await last_price(isin)
            except HTTPException:
                total += pos["qty"] * pos["entry"]
        return total
    cash = await tool("get_cash_balance", {})
    pf = await tool("get_portfolio", {})
    return float(cash.get("availableCash", 0)) + float(pf.get("netValue", 0))


async def execute(isin: str, side: str, qty: float, price: float, reason: str, mode: str = "market",
                  limit_price: float | None = None) -> dict:
    """Exécute (papier ou réel) un ordre, journalise, met à jour les stops."""
    if qty <= 0:
        raise HTTPException(400, "quantité nulle")
    r = state.data["risk"]
    notional = qty * price
    if side == "buy" and notional > float(r["capital"]) * float(r["max_position_pct"]) * 1.05:
        raise HTTPException(400, f"ordre refusé : {notional:.2f} EUR > plafond par position")
    name = await asset_name(isin)
    if state.data["dry_run"]:
        paper = state.data["paper"]
        pos = paper["positions"].get(isin, {"qty": 0.0, "entry": 0.0})
        if side == "buy":
            cost = notional + TR_FEE_PER_ORDER
            if cost > paper["cash"]:
                raise HTTPException(400, f"cash papier insuffisant ({paper['cash']:.2f} EUR)")
            paper["cash"] -= cost
            new_qty = pos["qty"] + qty
            pos["entry"] = (pos["entry"] * pos["qty"] + price * qty) / new_qty
            pos["qty"] = new_qty
            pos["name"] = name
            paper["positions"][isin] = pos
        else:
            qty = min(qty, pos["qty"])
            if qty <= 0:
                raise HTTPException(400, "aucune position papier à vendre")
            paper["cash"] += qty * price - TR_FEE_PER_ORDER
            pos["qty"] -= qty
            if pos["qty"] <= 1e-9:
                paper["positions"].pop(isin, None)
                state.data["stops"].pop(isin, None)
        result = {"orderId": f"paper-{len(state.data['journal'])+1}", "status": "simulated", "fee": TR_FEE_PER_ORDER}
    else:
        args = {"isin": isin, "exchange": "LSX", "orderType": side, "mode": mode, "size": qty, "expiry": "gfd"}
        if mode == "limit" and limit_price:
            args["limitPrice"] = limit_price
        result = await tool("place_order", args)
        if side == "sell":
            state.data["stops"].pop(isin, None)
    state.journal("order", f"{'PAPIER' if state.data['dry_run'] else 'RÉEL'} {side.upper()} {qty} x {name} @ {price:.2f} ({reason})",
                  isin=isin, side=side, qty=qty, price=price, result=result)
    return result


# --------------------------------------------------------------------------- autopilote
async def kill_switch_check() -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    eq = await equity()
    if state.data.get("day") != today:
        state.data["day"], state.data["day_start_equity"] = today, eq
    start = state.data["day_start_equity"] or eq
    dd = eq / start - 1 if start else 0.0
    if dd < -float(state.data["risk"]["max_daily_loss_pct"]):
        state.journal("kill", f"KILL SWITCH : {dd:.2%} de perte aujourd'hui. Liquidation et arrêt.")
        for isin, pos in list((await positions()).items()):
            try:
                await execute(isin, "sell", pos["qty"], await last_price(isin), "kill_switch")
            except HTTPException as exc:
                state.journal("error", f"liquidation {isin} impossible : {exc.detail}")
        state.data["halted"] = True
        state.data["autopilot"]["enabled"] = False
        state.save()
        return True
    return False


async def check_stops() -> list[str]:
    """Vérifie les stops de protection sur le dernier prix (appelé entre deux cycles)."""
    hits: list[str] = []
    held = await positions()
    for isin, pos in list(held.items()):
        stop = state.data["stops"].get(isin)
        if stop is None:
            continue
        try:
            price = await last_price(isin)
        except HTTPException:
            continue
        if price <= stop:
            await execute(isin, "sell", pos["qty"], price, f"stop {stop:.2f}")
            hits.append(isin)
    state.data["last_stop_check"] = datetime.now(timezone.utc).isoformat()
    state.save()
    return hits


async def run_cycle() -> dict:
    """Un cycle d'auto-trading complet sur toute la watchlist.

    1. kill switch (perte journalière) ; 2. pour chaque instrument : historique -> signal sur la
    dernière bougie CLOSE -> stop / trailing stop -> entrée (marché ouvert, taille fraction fixe
    avec frais TR, nb de positions max) ou sortie sur signal ; 3. état persisté et journalisé.
    """
    ap = state.data["autopilot"]
    rk = state.data["risk"]
    report: dict[str, Any] = {}
    if await kill_switch_check():
        return {"halted": True}
    held = await positions()
    eq = await equity()
    for item in state.data["watchlist"]:
        isin = item["isin"]
        name = item.get("name") or await asset_name(isin)
        st: dict[str, Any] = {"name": name, "ts": datetime.now(timezone.utc).isoformat()}
        try:
            df, raw = await history(isin, ap["range"])
            if len(df) < 60:
                st["status"] = "historique insuffisant"
                report[isin] = st; state.data["instruments"][isin] = st
                continue
            prepared, sig, _ = analyse_frame(df, ap["params"])
            signal = int(sig.iloc[-1])
            atr_now = float(prepared["atr"].iloc[-1])
            price = await last_price(isin)
            pos = held.get(isin)
            stop = state.data["stops"].get(isin)
            st.update({"signal": signal, "price": price, "atr": atr_now, "last_bar": str(df.index[-1])})

            # stop de protection / stop suiveur
            if pos and stop is not None and price <= stop:
                await execute(isin, "sell", pos["qty"], price, f"stop {stop:.2f}")
                pos = None
                held.pop(isin, None)
                st["status"] = "stop touché -> vendu"
            elif pos and rk.get("trailing_stop", True):
                new_stop = atr_stop(price, atr_now, float(rk["atr_stop_mult"]), 1)
                if stop is None or new_stop > stop:
                    state.data["stops"][isin] = new_stop
                    st["stop_moved"] = new_stop

            last_bar = str(df.index[-1])
            if item.get("last_bar") != last_bar:  # une seule décision par bougie close
                item["last_bar"] = last_bar
                if signal > 0 and not pos:
                    if len(held) >= int(rk.get("max_positions", 3)):
                        st["status"] = "signal achat mais nombre max de positions atteint"
                    else:
                        status = await tool("get_market_status", {"isin": isin, "exchange": "LSX"})
                        if not status.get("isOpen"):
                            st["status"] = f"signal achat mais marché {status.get('status', 'fermé')}"
                        else:
                            stop_lvl = atr_stop(price, atr_now, float(rk["atr_stop_mult"]), 1)
                            size = tr_position_size(eq, float(rk["risk_per_trade"]), price, stop_lvl,
                                                    fees_in_risk=bool(rk["fees_in_risk"]),
                                                    max_position_pct=float(rk["max_position_pct"]))
                            if size["qty"] <= 0:
                                state.journal("skip", f"{name} : signal achat mais taille nulle", warnings=size["warnings"])
                                st["status"] = "taille nulle : " + "; ".join(size["warnings"])
                            else:
                                await execute(isin, "buy", size["qty"], price, "signal")
                                state.data["stops"][isin] = stop_lvl
                                held[isin] = {"qty": size["qty"], "entry": price}
                                st["status"] = f"ACHAT {size['qty']} @ {price:.2f}, stop {stop_lvl:.2f}"
                elif signal <= 0 and pos:
                    await execute(isin, "sell", pos["qty"], price, "signal")
                    held.pop(isin, None)
                    st["status"] = f"VENTE sur signal @ {price:.2f}"
                else:
                    st["status"] = "en position, signal maintenu" if pos else "pas de signal"
            else:
                st.setdefault("status", "bougie déjà traitée" + (" (en position)" if pos else ""))
            st["held"] = isin in held
            st["stop"] = state.data["stops"].get(isin)
        except HTTPException as exc:
            st["status"] = f"erreur : {exc.detail}"
            if exc.status_code == 401:
                report[isin] = st; state.data["instruments"][isin] = st
                break
        report[isin] = st
        state.data["instruments"][isin] = st
    state.data["last_cycle"] = datetime.now(timezone.utc).isoformat()
    state.save()
    state.journal("cycle", "cycle autopilote terminé : " + ", ".join(f"{v.get('name', k)}: {v.get('status', '?')}" for k, v in report.items()))
    return report


async def autopilot_loop() -> None:
    while True:
        try:
            ap = state.data["autopilot"]
            if ap["enabled"] and not state.data["halted"]:
                last = state.data.get("last_cycle")
                due = last is None or (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() \
                    >= int(ap["interval_min"]) * 60
                if due:
                    await run_cycle()
                else:
                    last_chk = state.data.get("last_stop_check")
                    if last_chk is None or (datetime.now(timezone.utc) - datetime.fromisoformat(last_chk)).total_seconds() >= 300:
                        hits = await check_stops()
                        if hits:
                            state.journal("stop", "stops déclenchés entre deux cycles : " + ", ".join(hits))
        except Exception:
            log.exception("autopilote : erreur de cycle")
        await asyncio.sleep(30)


# --------------------------------------------------------------------------- API
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task = asyncio.create_task(autopilot_loop())
    yield
    task.cancel()
    await mcp.aclose() if hasattr(mcp, "aclose") else None


app = FastAPI(title="Tradebot — Trade Republic", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
async def api_status():
    tools: list[str] = []
    try:
        tools = await mcp.list_tools()
        if auth["status"] in ("unknown", "mcp_down"):
            auth["status"], auth["message"] = "unknown", ""
    except McpError as exc:
        auth["status"], auth["message"] = "mcp_down", str(exc)
    eq = None
    if state.data["dry_run"]:
        try:
            eq = await equity()
        except HTTPException:
            eq = None
    return clean({"mcp_url": MCP_URL, "mcp_tools": len(tools), "auth": auth, "dry_run": state.data["dry_run"],
                  "halted": state.data["halted"], "autopilot": state.data["autopilot"], "risk": state.data["risk"],
                  "watchlist": state.data["watchlist"], "equity": eq, "day_start_equity": state.data["day_start_equity"],
                  "last_cycle": state.data["last_cycle"], "last_stop_check": state.data.get("last_stop_check"),
                  "instruments": state.data.get("instruments", {}), "paper": state.data["paper"], "stops": state.data["stops"],
                  "fee_per_order": TR_FEE_PER_ORDER})


class Code(BaseModel):
    code: str = Field(min_length=4, max_length=8)


@app.post("/api/auth/2fa")
async def api_2fa(body: Code):
    res = await tool("enter_two_factor_code", {"code": body.code})
    auth["status"], auth["message"] = "ok", str(res.get("message", res) if isinstance(res, dict) else res)
    state.journal("auth", auth["message"])
    return {"message": auth["message"]}


@app.post("/api/auth/probe")
async def api_probe():
    """Déclenche l'authentification (envoi du SMS) en appelant un outil authentifié."""
    res = await tool("get_cash_balance", {})
    return clean(res)


@app.get("/api/journal")
async def api_journal(limit: int = 100):
    return state.data["journal"][:limit]


class Settings(BaseModel):
    dry_run: bool | None = None
    risk: dict | None = None
    autopilot: dict | None = None
    halted: bool | None = None
    paper_reset: float | None = None


@app.post("/api/settings")
async def api_settings(body: Settings):
    if body.risk is not None:
        state.data["risk"].update(body.risk)
        risk_cfg().validate()
    if body.autopilot is not None:
        state.data["autopilot"].update(body.autopilot)
    if body.dry_run is not None and body.dry_run != state.data["dry_run"]:
        state.data["dry_run"] = body.dry_run
        state.journal("mode", "passage en mode " + ("PAPIER" if body.dry_run else "RÉEL (ordres envoyés à Trade Republic)"))
    if body.halted is not None:
        state.data["halted"] = body.halted
        if not body.halted:
            state.journal("mode", "kill switch réarmé manuellement")
    if body.paper_reset is not None:
        state.data["paper"] = {"cash": float(body.paper_reset), "positions": {}}
        state.data["stops"] = {}
        state.data["day_start_equity"] = None
        state.journal("mode", f"portefeuille papier réinitialisé à {body.paper_reset:.2f} EUR")
    state.save()
    return await api_status()


class WatchItem(BaseModel):
    isin: str = Field(min_length=12, max_length=12)
    name: str | None = None


@app.post("/api/watchlist")
async def api_watch_add(body: WatchItem):
    if not any(w["isin"] == body.isin for w in state.data["watchlist"]):
        state.data["watchlist"].append({"isin": body.isin, "name": body.name or await asset_name(body.isin)})
        state.save()
    return state.data["watchlist"]


@app.delete("/api/watchlist/{isin}")
async def api_watch_del(isin: str):
    state.data["watchlist"] = [w for w in state.data["watchlist"] if w["isin"] != isin]
    state.save()
    return state.data["watchlist"]


@app.get("/api/search")
async def api_search(q: str, limit: int = 10):
    return clean(await tool("search_assets", {"query": q, "limit": limit}))


@app.get("/api/portfolio")
async def api_portfolio():
    out: dict[str, Any] = {"dry_run": state.data["dry_run"]}
    if state.data["dry_run"]:
        rows = []
        for isin, pos in state.data["paper"]["positions"].items():
            try:
                px = await last_price(isin)
            except HTTPException:
                px = pos["entry"]
            rows.append({"isin": isin, "name": pos.get("name") or await asset_name(isin), "qty": pos["qty"],
                         "entry": pos["entry"], "price": px, "value": pos["qty"] * px,
                         "pnl": (px - pos["entry"]) * pos["qty"], "stop": state.data["stops"].get(isin)})
        out.update({"cash": state.data["paper"]["cash"], "positions": rows,
                    "equity": state.data["paper"]["cash"] + sum(r["value"] for r in rows)})
    else:
        cash = await tool("get_cash_balance", {})
        pf = await tool("get_portfolio", {})
        rows = []
        for p in pf.get("positions", []):
            qty = float(p["netSize"])
            val = float(p.get("netValue") or 0)
            rows.append({"isin": p["instrumentId"], "name": await asset_name(p["instrumentId"]), "qty": qty,
                         "entry": p.get("averageCost"), "price": val / qty if qty else None, "value": val,
                         "pnl": val - float(p.get("averageCost") or 0) * qty, "stop": state.data["stops"].get(p["instrumentId"])})
        out.update({"cash": cash.get("availableCash"), "currency": cash.get("currency"), "positions": rows,
                    "equity": float(cash.get("availableCash", 0)) + float(pf.get("netValue", 0)),
                    "unrealised": pf.get("unrealisedProfit")})
    return clean(out)


@app.get("/api/instrument/{isin}")
async def api_instrument(isin: str, range: str = "1y"):
    df, raw = await history(isin, range)
    if df.empty:
        raise HTTPException(404, "pas d'historique")
    params = state.data["autopilot"]["params"]
    prepared, sig, _ = analyse_frame(df, params)
    ppy = bars_per_year(raw, df)
    r = ind.log_returns(df["close"]).dropna()
    price = None
    try:
        price = await tool("get_price", {"isin": isin, "exchange": "LSX"})
    except HTTPException:
        pass
    analysis = None
    try:
        analysis = await mcp.call("get_detailed_analysis", {"isin": isin, "range": range if range != "1d" else "3m"})
    except Exception as exc:
        analysis = {"error": str(exc)[:200]}
    status = None
    try:
        status = await mcp.call("get_market_status", {"isin": isin, "exchange": "LSX"})
    except Exception:
        pass
    cols = ["close", "ema_fast", "ema_slow", "bb_upper", "bb_lower", "rsi", "atr", "macd_hist"]
    series = {c: clean(prepared[c].round(4)) for c in cols if c in prepared}
    last = prepared.iloc[-1]
    atr_now = float(last["atr"]) if not np.isnan(last["atr"]) else None
    px = float(price.get("last") or price.get("ask")) if price else float(last["close"])
    rk = state.data["risk"]
    stop_lvl = atr_stop(px, atr_now, float(rk["atr_stop_mult"]), 1) if atr_now else None
    sizing = tr_position_size(float(rk["capital"]), float(rk["risk_per_trade"]), px, stop_lvl,
                              fees_in_risk=bool(rk["fees_in_risk"]), max_position_pct=float(rk["max_position_pct"])) if stop_lvl else None
    return clean({
        "isin": isin, "name": await asset_name(isin), "range": range, "bars": len(df), "bars_per_year": ppy,
        "ts": [t.isoformat() for t in prepared.index], "series": series, "signal": clean(sig),
        "current_signal": int(sig.iloc[-1]), "price": price, "market": status, "analysis": analysis,
        "stats": {"rsi": last.get("rsi"), "atr": atr_now, "atr_pct": (atr_now / px) if atr_now else None,
                  "vol_annual": float(r.std() * np.sqrt(ppy)) if len(r) > 2 else None,
                  "var_95": var_historical(r), "es_95": expected_shortfall(r),
                  "ret_1m": float(df["close"].iloc[-1] / df["close"].iloc[-min(21, len(df))] - 1)},
        "suggestion": {"stop": stop_lvl, "sizing": sizing},
    })


@app.get("/api/instrument/{isin}/news")
async def api_news(isin: str):
    out: dict[str, Any] = {}
    for key, tl, args in (("news", "get_news", {"isin": isin}), ("sentiment", "get_sentiment", {"isin": isin}),
                          ("fundamentals", "get_fundamentals", {"isin": isin})):
        try:
            out[key] = await mcp.call(tl, args)
        except Exception as exc:
            out[key] = {"error": str(exc)[:300]}
    return clean(out)


class BacktestReq(BaseModel):
    isin: str
    range: str = "1y"
    params: dict = Field(default_factory=dict)
    walkforward: bool = False
    train_bars: int = 120
    test_bars: int = 40


@app.post("/api/backtest")
async def api_backtest(body: BacktestReq):
    df, raw = await history(body.isin, body.range)
    if len(df) < 80:
        raise HTTPException(400, f"seulement {len(df)} barres : élargissez la plage (1y, 5y, max)")
    params = {**state.data["autopilot"]["params"], **body.params}
    ppy = bars_per_year(raw, df)
    risk = risk_cfg()
    # Trade Republic : 1 EUR fixe par ordre ~ on l'approxime par un taux sur le notional moyen engagé
    avg_notional = max(risk.capital * risk.max_position_pct, 1.0)
    risk.fee_rate = TR_FEE_PER_ORDER / avg_notional
    prepared, sig, strat = analyse_frame(df, params)
    res = run_backtest(prepared, sig, risk, ppy)
    out: dict[str, Any] = {
        "isin": body.isin, "name": await asset_name(body.isin), "bars": len(df), "params": strat.params,
        "fee_rate_equiv": risk.fee_rate,
        "metrics": res.metrics, "ts": [t.isoformat() for t in res.equity.index],
        "equity": clean(res.equity.round(4)), "buy_hold": clean((risk.capital * df["close"] / df["close"].iloc[0]).round(4)),
        "trades": clean(res.trades.to_dict(orient="records")),
    }
    if body.walkforward:
        try:
            wf = walk_forward(df, state.data["autopilot"]["strategy"], risk, ppy, train_bars=body.train_bars,
                              test_bars=body.test_bars, base_params=params,
                              param_grid={"ema_fast": [10, 20, 30], "min_votes": [3, 4]})
            out["walkforward"] = {"metrics": wf.metrics, "windows": clean(wf.windows.to_dict(orient="records")),
                                  "ts": [t.isoformat() for t in wf.oos_equity.index], "equity": clean(wf.oos_equity.round(4))}
        except ValueError as exc:
            out["walkforward"] = {"error": str(exc)}
    return clean(out)


class OrderReq(BaseModel):
    isin: str
    side: str
    qty: float
    mode: str = "market"
    limit_price: float | None = None
    confirm: bool = False


@app.post("/api/orders")
async def api_place_order(body: OrderReq):
    if not body.confirm:
        raise HTTPException(400, "confirm=true requis")
    if body.side not in ("buy", "sell"):
        raise HTTPException(400, "side doit être buy ou sell")
    price = body.limit_price or await last_price(body.isin)
    return clean(await execute(body.isin, body.side, body.qty, price, "manuel", body.mode, body.limit_price))


@app.get("/api/orders")
async def api_orders():
    if state.data["dry_run"]:
        return {"dry_run": True, "orders": [j for j in state.data["journal"] if j["kind"] == "order"][:50]}
    return clean(await tool("get_orders", {"includeExecuted": True, "includeCancelled": False}))


@app.delete("/api/orders/{order_id}")
async def api_cancel(order_id: str):
    if state.data["dry_run"]:
        raise HTTPException(400, "pas d'ordre en attente en mode papier")
    return clean(await tool("cancel_order", {"orderId": order_id}))


class PlanReq(BaseModel):
    isin: str
    amount: float = Field(gt=0)
    interval: str = "monthly"
    start_date: str | None = None
    confirm: bool = False


@app.get("/api/savings-plans")
async def api_plans():
    return clean(await tool("get_savings_plans", {}))


@app.post("/api/savings-plans")
async def api_plan_create(body: PlanReq):
    if not body.confirm:
        raise HTTPException(400, "confirm=true requis")
    if state.data["dry_run"]:
        state.journal("plan", f"PAPIER plan d'épargne {body.amount:.2f} EUR {body.interval} sur {body.isin}")
        return {"status": "simulated"}
    args = {"isin": body.isin, "amount": body.amount, "interval": body.interval}
    if body.start_date:
        args["startDate"] = body.start_date
    res = await tool("create_savings_plan", args)
    state.journal("plan", f"RÉEL plan d'épargne {body.amount:.2f} EUR {body.interval} sur {body.isin}", result=res)
    return clean(res)


@app.delete("/api/savings-plans/{plan_id}")
async def api_plan_cancel(plan_id: str):
    if state.data["dry_run"]:
        raise HTTPException(400, "mode papier : aucun plan réel")
    return clean(await tool("cancel_savings_plan", {"id": plan_id}))


@app.post("/api/autopilot/run")
async def api_run_now():
    if state.data["halted"]:
        raise HTTPException(400, "kill switch actif : réarmer d'abord")
    return clean(await run_cycle())
