from __future__ import annotations

import uvicorn

from src.api import create_app
from src.core.config import load_config


def main() -> None:
    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(
        app,
        host=cfg.api.host,
        port=cfg.api.port,
        reload=cfg.api.reload,
    )


if __name__ == "__main__":
    main()
