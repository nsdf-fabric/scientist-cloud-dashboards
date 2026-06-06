#!/usr/bin/env python3
"""Run the Bokeh dashboard and FastAPI refresh endpoint together."""
from __future__ import annotations

import os
import signal
import threading
from typing import Optional

import uvicorn
from bokeh.application import Application
from bokeh.application.handlers import ScriptHandler
from bokeh.server.server import Server
from tornado.ioloop import IOLoop

from ornl_chess_strain_lib import load_simple_env_file
from refresh_api import app as refresh_app


def _env_value(name: str, default: str = "") -> str:
    env_file_values = load_simple_env_file(os.environ.get("ORNL_S3_ENV_FILE", "").strip())
    return (os.environ.get(name) or env_file_values.get(name) or default).strip()


def _int_env(name: str, default: int) -> int:
    try:
        value = int(_env_value(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def run_fastapi(stop_event: threading.Event) -> None:
    config = uvicorn.Config(
        refresh_app,
        host=_env_value("ORNL_REFRESH_API_HOST", "0.0.0.0"),
        port=_int_env("ORNL_REFRESH_API_PORT", 8060),
        log_level=_env_value("ORNL_REFRESH_API_LOG_LEVEL", "info"),
    )
    server = uvicorn.Server(config)

    def _watch_stop() -> None:
        stop_event.wait()
        server.should_exit = True

    watcher = threading.Thread(target=_watch_stop, daemon=True)
    watcher.start()
    server.run()


def main() -> None:
    stop_event = threading.Event()
    api_thread = threading.Thread(target=run_fastapi, args=(stop_event,), daemon=True)
    api_thread.start()

    bokeh_port = _int_env("ORNL_BOKEH_PORT", 8059)
    bokeh_host = _env_value("ORNL_BOKEH_HOST", "0.0.0.0")
    websocket_origin = _env_value("ORNL_BOKEH_ALLOW_WEBSOCKET_ORIGIN", "*")
    io_loop = IOLoop.current()
    app = Application(ScriptHandler(filename="ORNL_CHESS_strain.py"))
    server: Optional[Server] = Server(
        {"/ORNL_CHESS_strain": app},
        io_loop=io_loop,
        port=bokeh_port,
        address=bokeh_host,
        allow_websocket_origin=[websocket_origin],
    )

    def _shutdown(signum: int, frame: object) -> None:
        stop_event.set()
        if server is not None:
            server.stop()
        io_loop.add_callback(io_loop.stop)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.start()
    print(
        f"Bokeh dashboard: http://{bokeh_host}:{bokeh_port}/ORNL_CHESS_strain/",
        flush=True,
    )
    print(
        "FastAPI refresh endpoint: "
        f"http://{_env_value('ORNL_REFRESH_API_HOST', '0.0.0.0')}:"
        f"{_int_env('ORNL_REFRESH_API_PORT', 8060)}/refresh",
        flush=True,
    )
    io_loop.start()
    stop_event.set()
    api_thread.join(timeout=5)


if __name__ == "__main__":
    main()
