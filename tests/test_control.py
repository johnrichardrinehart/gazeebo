"""Tests for process-scoped on-demand training control."""

from __future__ import annotations

import asyncio
import stat
import tempfile
import unittest
from pathlib import Path

from gazeebo.control import ControlError, TrainingControl, request_training


class TrainingControlTests(unittest.TestCase):
    """Lock request delivery, permissions, validation, and cleanup."""

    def test_active_process_accepts_request_and_removes_socket(self) -> None:
        """A second command can transition one foreground process into training."""

        async def exercise(root: Path) -> None:
            event = asyncio.Event()
            path = root / "gazeebo" / "control.sock"
            control = TrainingControl(event, path)
            await control.start()
            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert await request_training(path)
            assert event.is_set()
            await control.close()
            await control.close()
            assert not path.exists()

        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(exercise(Path(temporary)))

    def test_requesting_process_cannot_unlink_owner_socket(self) -> None:
        """A short-lived command closes only control paths that it created."""

        async def exercise(root: Path) -> None:
            event = asyncio.Event()
            path = root / "gazeebo" / "control.sock"
            owner = TrainingControl(event, path)
            requester = TrainingControl(asyncio.Event(), path)
            await owner.start()
            assert await request_training(path)
            await requester.close()
            assert path.exists()
            event.clear()
            assert await request_training(path)
            assert event.is_set()
            await owner.close()
            assert not path.exists()

        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(exercise(Path(temporary)))

    def test_absent_process_returns_false_without_creating_files(self) -> None:
        """A training command can become the foreground owner when no socket exists."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gazeebo" / "control.sock"
            assert not asyncio.run(request_training(path))
            assert not path.exists()

    def test_non_socket_control_path_fails_without_deleting_it(self) -> None:
        """An attacker-controlled regular file is never followed or removed."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "control.sock"
            path.write_text("keep", encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(ControlError, "socket"):
                asyncio.run(request_training(path))
            assert path.read_text(encoding="utf-8") == "keep"

    def test_shared_runtime_directory_is_rejected(self) -> None:
        """Other local users cannot access the process control channel."""

        async def exercise(path: Path) -> None:
            control = TrainingControl(asyncio.Event(), path)
            with self.assertRaisesRegex(ControlError, "owner-only"):
                await control.start()

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "gazeebo"
            directory.mkdir(mode=0o700)
            directory.chmod(0o755)
            asyncio.run(exercise(directory / "control.sock"))
            directory.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
