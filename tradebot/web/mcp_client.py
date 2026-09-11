"""Client minimal du serveur MCP Trade Republic (transport « Streamable HTTP », JSON-RPC 2.0).

Le serveur (vendor/trade-republic-mcp-server) est sans état : chaque POST /mcp
crée une instance de serveur, on peut donc appeler `tools/call` directement.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx


class McpError(Exception):
    """Erreur renvoyée par un outil MCP (isError=true) ou par le transport."""


class TwoFactorRequired(McpError):
    """Trade Republic demande un code 2FA (SMS/app) : appeler enter_two_factor_code."""


_2FA_PATTERNS = re.compile(r"two.?factor|2fa|TwoFactorCodeRequired", re.IGNORECASE)


class McpClient:
    def __init__(self, url: str = "http://localhost:3006/mcp", timeout: float = 90.0):
        self.url = url
        self.timeout = timeout
        self._id = 0
        self._client = httpx.AsyncClient(timeout=timeout, headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        })

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _rpc(self, method: str, params: dict | None = None) -> Any:
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        try:
            resp = await self._client.post(self.url, content=json.dumps(payload))
        except httpx.HTTPError as exc:
            raise McpError(f"serveur MCP injoignable ({self.url}) : {exc}") from exc
        if resp.status_code >= 400:
            raise McpError(f"HTTP {resp.status_code} : {resp.text[:300]}")
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            msg = self._parse_sse(resp.text, self._id)
        else:
            msg = resp.json()
        if "error" in msg:
            raise McpError(msg["error"].get("message", str(msg["error"])))
        return msg.get("result")

    @staticmethod
    def _parse_sse(text: str, want_id: int) -> dict:
        last = None
        for block in text.split("\n\n"):
            data = "".join(line[5:].strip() for line in block.splitlines() if line.startswith("data:"))
            if not data:
                continue
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("id") == want_id:
                return msg
            last = msg
        if last is None:
            raise McpError("réponse SSE vide")
        return last

    async def list_tools(self) -> list[str]:
        res = await self._rpc("tools/list")
        return [t["name"] for t in res.get("tools", [])]

    async def call(self, tool: str, arguments: dict | None = None) -> Any:
        """Appelle un outil et renvoie son résultat JSON désérialisé (ou le texte brut)."""
        res = await self._rpc("tools/call", {"name": tool, "arguments": arguments or {}})
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if res.get("isError"):
            if _2FA_PATTERNS.search(text):
                raise TwoFactorRequired(text)
            raise McpError(text or f"{tool} a échoué")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
