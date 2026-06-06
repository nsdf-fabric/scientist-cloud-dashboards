"""FastAPI endpoint that asks active Bokeh dashboard sessions to refresh."""
from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI, Header, HTTPException, status

from ornl_chess_strain_lib import load_simple_env_file
from refresh_bus import registered_count, trigger_refresh


def _env_value(name: str) -> str:
    env_file_values = load_simple_env_file(os.environ.get("ORNL_S3_ENV_FILE", "").strip())
    return (os.environ.get(name) or env_file_values.get(name) or "").strip()


def expected_api_key() -> str:
    return _env_value("ORNL_REFRESH_API_KEY")


def create_app() -> FastAPI:
    app = FastAPI(title="DIAL Dashboard Refresh API")

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {"ok": True, "registered_sessions": registered_count()}

    @app.post("/refresh")
    def refresh(x_api_key: str = Header(default="", alias="X-API-Key")) -> Dict[str, object]:
        api_key = expected_api_key()
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="ORNL_REFRESH_API_KEY is not configured.",
            )
        if x_api_key != api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key.",
            )
        triggered = trigger_refresh()
        return {"ok": True, "triggered": triggered}

    return app


app = create_app()
