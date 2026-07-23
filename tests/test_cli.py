"""Tests for the command-line boundary."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from gazeebo.cli import _camera_device, _open_startup_resources, build_parser
from gazeebo.game import GameConfig


class CliTests(unittest.TestCase):
    """Lock argument parsing without opening runtime resources."""

    def test_defaults_define_one_local_foreground_session(self) -> None:
        """An empty argument list selects safe runtime defaults."""
        arguments = build_parser().parse_args([])
        assert arguments.command == "run"
        assert arguments.camera is None
        assert arguments.width == 640
        assert arguments.height == 480
        assert arguments.calibration_samples > 0
        assert 0.0 < arguments.open_eye_threshold < 1.0
        assert arguments.pointer_update_interval == 0.10
        assert not arguments.ephemeral
        assert arguments.game_batch_size == 5
        assert arguments.game_precision_threshold == 100.0
        assert arguments.game_maximum_targets == 55
        assert arguments.game_settle + arguments.game_dwell == 2.0
        assert not arguments.debug_hud

    def test_training_commands_do_not_expose_profiles(self) -> None:
        """Users request training or reset one automatic local corpus."""
        train = build_parser().parse_args(["train"])
        reset = build_parser().parse_args(["reset-training"])
        assert train.command == "train"
        assert reset.command == "reset-training"
        assert "profile" not in vars(train)

    def test_camera_index_and_path_remain_distinct(self) -> None:
        """Numeric indices and device paths reach OpenCV in their native forms."""
        assert _camera_device(None) is None
        assert _camera_device("2") == 2
        assert _camera_device("/dev/video2") == "/dev/video2"

    def test_zero_pointer_interval_requests_continuous_updates(self) -> None:
        """A zero interval remains available for explicit development tuning."""
        arguments = build_parser().parse_args(["--pointer-update-interval", "0"])
        assert arguments.pointer_update_interval == 0.0

    def test_startup_stop_closes_parallel_camera_resources(self) -> None:
        """A signal during portal authorization cannot leak camera or vision state."""

        class Resource:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        camera = Resource()
        vision = Resource()
        portal_started = asyncio.Event()

        async def authorize() -> None:
            portal_started.set()
            await asyncio.Event().wait()

        async def scenario() -> object:
            stop = asyncio.Event()
            task = asyncio.create_task(_open_startup_resources(build_parser().parse_args([]), stop))
            await portal_started.wait()
            stop.set()
            return await task

        with (
            patch("gazeebo.cli.PortalPointerController.authorize", side_effect=authorize),
            patch("gazeebo.cli._open_vision", return_value=(camera, vision)),
        ):
            assert asyncio.run(scenario()) is None
        assert camera.closed
        assert vision.closed

    def test_non_terminating_game_values_are_rejected(self) -> None:
        """Invalid game timing fails before runtime resources can open."""
        arguments = build_parser().parse_args(["--game-dwell", "2", "--game-target-timeout", "1"])
        with self.assertRaisesRegex(ValueError, "timeout"):
            GameConfig(
                batch_size=arguments.game_batch_size,
                precision_threshold=arguments.game_precision_threshold,
                maximum_targets=arguments.game_maximum_targets,
                settle_seconds=arguments.game_settle,
                dwell_seconds=arguments.game_dwell,
                target_timeout_seconds=arguments.game_target_timeout,
                minimum_diameter=arguments.game_minimum_diameter,
                maximum_diameter=arguments.game_maximum_diameter,
            )


if __name__ == "__main__":
    unittest.main()
