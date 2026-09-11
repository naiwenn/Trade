# Image unique : serveur MCP (Node) + interface web (Python), supervisés par start.sh
FROM node:22-bookworm-slim AS mcp
WORKDIR /app/vendor/trade-republic-mcp-server
COPY vendor/trade-republic-mcp-server/package*.json ./
RUN npm ci --no-audit --no-fund
COPY vendor/trade-republic-mcp-server/ ./
RUN npm run build && npm prune --omit=dev

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY tradebot ./tradebot
COPY scripts ./scripts
COPY config.yaml ./
COPY --from=mcp /app/vendor/trade-republic-mcp-server/dist ./vendor/trade-republic-mcp-server/dist
COPY --from=mcp /app/vendor/trade-republic-mcp-server/node_modules ./vendor/trade-republic-mcp-server/node_modules
COPY vendor/trade-republic-mcp-server/package.json ./vendor/trade-republic-mcp-server/
RUN mkdir -p data && touch vendor/trade-republic-mcp-server/.env
ENV WEB_HOST=0.0.0.0 WEB_PORT=8080 MCP_PORT=3006
EXPOSE 8080
CMD ["./scripts/start.sh"]
