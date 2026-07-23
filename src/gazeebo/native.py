"""Load the packaged in-process Wayland renderer."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


class NativeRendererError(RuntimeError):
    """The packaged Wayland renderer cannot be loaded."""


def load_native_renderer() -> ctypes.CDLL:
    """Load the shared renderer used by opt-in graphical diagnostics."""
    library_path = os.environ.get("GAZEEBO_HUD_LIBRARY")
    if not library_path or not Path(library_path).is_file():
        msg = "packaged Wayland renderer is unavailable"
        raise NativeRendererError(msg)
    try:
        return ctypes.CDLL(library_path)
    except OSError as load_error:
        msg = "packaged Wayland renderer could not be loaded"
        raise NativeRendererError(msg) from load_error
