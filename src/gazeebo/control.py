"""Owner-only runtime control for on-demand training transitions."""

from __future__ import annotations

import asyncio
import contextlib
import os
import stat
from pathlib import Path

_DIRECTORY_MODE = 0o700
_SOCKET_MODE = 0o600


class ControlError(RuntimeError):
    """The local training-control channel is unsafe or unavailable."""


class TrainingControl:
    """Own one process-scoped Unix socket that accepts a training request."""

    def __init__(
        self,
        training_requested: asyncio.Event,
        path: Path | None = None,
    ) -> None:
        """Bind one training event to an optional testable socket path."""
        self.training_requested = training_requested
        self.path = path or control_path()
        self._server: asyncio.Server | None = None
        self._owns_path = False

    async def start(self) -> None:
        """Create one owner-only socket for the foreground process."""
        directory = self.path.parent
        if not directory.exists():
            directory.mkdir(mode=_DIRECTORY_MODE, parents=True)
            directory.chmod(_DIRECTORY_MODE)
        _validate_directory(directory)
        if self.path.exists() or self.path.is_symlink():
            msg = "another Gazeebo control socket already exists"
            raise ControlError(msg)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle,
                path=self.path,
            )
            self._owns_path = True
            self.path.chmod(_SOCKET_MODE)
            _validate_socket(self.path)
        except OSError as error:
            msg = "could not create the training control socket"
            raise ControlError(msg) from error

    async def close(self) -> None:
        """Close the server and remove its socket idempotently."""
        server, self._server = self._server, None
        owns_path, self._owns_path = self._owns_path, False
        if server is not None:
            server.close()
            await server.wait_closed()
        if owns_path and (self.path.exists() or self.path.is_symlink()):
            try:
                _validate_socket(self.path)
                self.path.unlink()
            except FileNotFoundError:
                pass
        with contextlib.suppress(OSError):
            self.path.parent.rmdir()

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            command = await asyncio.wait_for(reader.readline(), 2.0)
            if command == b"train\n":
                self.training_requested.set()
                writer.write(b"accepted\n")
            else:
                writer.write(b"rejected\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


async def request_training(path: Path | None = None) -> bool:
    """Request training from an existing process, returning false when absent."""
    target = path or control_path()
    if not target.exists() and not target.is_symlink():
        return False
    _validate_socket(target)
    try:
        reader, writer = await asyncio.open_unix_connection(target)
    except (ConnectionRefusedError, FileNotFoundError):
        _validate_socket(target)
        target.unlink()
        return False
    try:
        writer.write(b"train\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), 2.0)
        return response == b"accepted\n"
    finally:
        writer.close()
        await writer.wait_closed()


def control_path() -> Path:
    """Return the standard process-scoped runtime socket path."""
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        msg = "XDG_RUNTIME_DIR is required for on-demand training control"
        raise ControlError(msg)
    return Path(runtime) / "gazeebo" / "control.sock"


def _validate_directory(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        msg = "training control directory must be owner-only"
        raise ControlError(msg)


def _validate_socket(path: Path) -> None:
    metadata = path.lstat()
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        msg = "training control path must be an owner-only socket"
        raise ControlError(msg)
