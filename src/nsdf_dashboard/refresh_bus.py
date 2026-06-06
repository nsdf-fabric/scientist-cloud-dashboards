"""In-memory refresh coordination between FastAPI and Bokeh sessions."""
from __future__ import annotations

import itertools
import threading
from typing import Callable, Dict

RefreshCallback = Callable[[], None]

_LOCK = threading.Lock()
_NEXT_ID = itertools.count(1)
_CALLBACKS: Dict[int, RefreshCallback] = {}


def register_refresh_callback(callback: RefreshCallback) -> int:
    """Register one dashboard session callback and return its token."""
    token = next(_NEXT_ID)
    with _LOCK:
        _CALLBACKS[token] = callback
    return token


def unregister_refresh_callback(token: int) -> None:
    with _LOCK:
        _CALLBACKS.pop(token, None)


def trigger_refresh() -> int:
    """Trigger all registered dashboard callbacks. Returns callback count."""
    with _LOCK:
        callbacks = list(_CALLBACKS.values())
    for callback in callbacks:
        callback()
    return len(callbacks)


def registered_count() -> int:
    with _LOCK:
        return len(_CALLBACKS)


def clear_refresh_callbacks() -> None:
    with _LOCK:
        _CALLBACKS.clear()
