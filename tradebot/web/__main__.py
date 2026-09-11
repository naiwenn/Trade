import os

import uvicorn

uvicorn.run("tradebot.web.app:app", host=os.environ.get("WEB_HOST", "127.0.0.1"),
            port=int(os.environ.get("WEB_PORT", "8080")), log_level="info")
