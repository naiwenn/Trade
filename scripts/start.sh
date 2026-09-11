#!/usr/bin/env bash
# Démarre le serveur MCP Trade Republic (Node) puis l'interface web (Python).
# Usage : ./scripts/start.sh            (arrêt : Ctrl+C, les deux processus sont tués)
set -euo pipefail
cd "$(dirname "$0")/.."
MCP_DIR=vendor/trade-republic-mcp-server
export MCP_PORT="${MCP_PORT:-3006}"
export WEB_PORT="${WEB_PORT:-8080}"
export WEB_HOST="${WEB_HOST:-127.0.0.1}"
export MCP_URL="http://127.0.0.1:${MCP_PORT}/mcp"

if [ ! -f "$MCP_DIR/package.json" ]; then
  echo "Sous-module absent : git submodule update --init" >&2; exit 1
fi
if [ ! -f "$MCP_DIR/.env" ]; then
  echo "Créez $MCP_DIR/.env avec TRADE_REPUBLIC_PHONE_NUMBER et TRADE_REPUBLIC_PIN (voir $MCP_DIR/.env.example)" >&2; exit 1
fi
if [ ! -d "$MCP_DIR/node_modules" ]; then (cd "$MCP_DIR" && npm ci --no-audit --no-fund); fi
if [ ! -f "$MCP_DIR/dist/index.js" ]; then (cd "$MCP_DIR" && npm run build); fi

(cd "$MCP_DIR" && PORT="$MCP_PORT" node dist/index.js) &
MCP_PID=$!
trap 'kill $MCP_PID 2>/dev/null || true' EXIT INT TERM
sleep 2
echo "Interface : http://${WEB_HOST}:${WEB_PORT}"
python3 -m uvicorn tradebot.web.app:app --host "$WEB_HOST" --port "$WEB_PORT"
