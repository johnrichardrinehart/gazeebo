"""Tests for foreground session lifecycle and cleanup."""

from __future__ import annotations

import asyncio
import unittest
from collections import deque

from gazeebo.contracts import DisplayRegion, EyeObservation, RuntimeStatus
from gazeebo.geometry import DisplayTopology, calibration_targets
from gazeebo.runtime import TrackingConfig, run_owned_session
from tests.fakes import (
    FakeCamera,
    FakeGame,
    FakeHud,
    FakePointer,
    FakeStatus,
    FakeVision,
)


def observation(index: float) -> EyeObservation:
    """Create an open-eye calibration feature."""
    value = float(index)
    return EyeObservation(value, 1.0, 1.0, (value,), 1.0, (0.0, 0.5))


def fast_tracking() -> TrackingConfig:
    """Use one deterministic observation per calibration target."""
    return TrackingConfig(
        calibration_settle_seconds=0.0,
        calibration_samples_per_target=1,
        calibration_attempts_per_target=1,
        frame_interval_seconds=0.0,
        maximum_missed_frames=2,
    )


class RuntimeTests(unittest.TestCase):
    """Lock normal, failed, and topology-invalidated cleanup."""

    def test_tracking_stops_and_releases_every_owned_resource(self) -> None:
        """A stop event exits with no camera, model, or pointer owner left open."""
        camera = FakeCamera(deque(object() for _ in range(6)))
        vision = FakeVision(deque(observation(index) for index in range(6)))
        pointer = FakePointer((DisplayRegion("only", 100, 200, 1000, 700),))
        hud = FakeHud()
        status = FakeStatus()
        stop = asyncio.Event()

        async def controlled_sleep(_delay: float) -> None:
            if status.reports and status.reports[-1][0] is RuntimeStatus.ACTIVE:
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                stop,
                hud=hud,
                tracking=fast_tracking(),
                sleep=controlled_sleep,
            )
        )
        assert result == 0
        assert len(pointer.moves) == 6
        assert camera.closed
        assert vision.closed
        assert pointer.closed
        assert hud.closed
        assert hud.updates[-1][0] == "only"
        assert hud.updates[-1][1] >= 100.0
        assert hud.updates[-1][2] >= 200.0
        assert [item[0] for item in status.reports][-2:] == [
            RuntimeStatus.ACTIVE,
            RuntimeStatus.STOPPED,
        ]

    def test_calibration_and_tracking_cross_authorized_displays(self) -> None:
        """One fitted session can move between every authorized region."""
        regions = (
            DisplayRegion("left", 0, 0, 1000, 700),
            DisplayRegion("right", 1000, 0, 1000, 700),
        )
        topology = DisplayTopology(regions)
        targets = calibration_targets(topology)
        observations: deque[EyeObservation | None] = deque(
            EyeObservation(
                float(index),
                1.0,
                1.0,
                (
                    topology.to_global(target).x,
                    topology.to_global(target).y,
                ),
                1.0,
                (0.0, 0.5),
            )
            for index, target in enumerate(targets)
        )
        observations.append(EyeObservation(20.0, 1.0, 1.0, (1500.0, 350.0), 1.0, (0.0, 0.5)))
        pointer = FakePointer(regions)
        status = FakeStatus()
        stop = asyncio.Event()

        async def stop_after_tracking_move(_delay: float) -> None:
            if (
                status.reports
                and status.reports[-1][0] is RuntimeStatus.ACTIVE
                and len(pointer.moves) > len(targets)
            ):
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                FakeCamera(deque(object() for _ in observations)),
                FakeVision(observations),
                pointer,
                status,
                stop,
                tracking=fast_tracking(),
                sleep=stop_after_tracking_move,
            )
        )
        assert result == 0
        assert {move[0] for move in pointer.moves[: len(targets)]} == {"left", "right"}
        assert pointer.moves[-1][0] == "right"

    def test_pointer_updates_are_rate_limited_without_throttling_observations(self) -> None:
        """The default cadence emits at most ten pointer moves per second."""
        timestamps = (0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 10.02, 10.09, 10.11)
        camera = FakeCamera(deque(object() for _ in timestamps))
        vision = FakeVision(deque(observation(value) for value in timestamps))
        pointer = FakePointer((DisplayRegion("only", 0, 0, 1000, 700),))
        status = FakeStatus()
        stop = asyncio.Event()

        async def stop_after_second_tracking_move(_delay: float) -> None:
            if (
                status.reports
                and status.reports[-1][0] is RuntimeStatus.ACTIVE
                and len(pointer.moves) == 7
            ):
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                stop,
                tracking=fast_tracking(),
                sleep=stop_after_second_tracking_move,
            )
        )
        assert result == 0
        assert len(pointer.moves) == 7
        assert len(vision.observations) == 0

    def test_calibration_failure_still_releases_resources(self) -> None:
        """An exhausted camera cannot bypass the shared cleanup path."""
        camera = FakeCamera(deque())
        vision = FakeVision(deque())
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        status = FakeStatus()
        game = FakeGame()

        with self.assertRaisesRegex(EOFError, "exhausted"):
            asyncio.run(
                run_owned_session(
                    camera,
                    vision,
                    pointer,
                    status,
                    asyncio.Event(),
                    game=game,
                    tracking=fast_tracking(),
                )
            )
        assert camera.closed
        assert vision.closed
        assert pointer.closed
        assert game.closed
        assert status.reports[-1][0] is RuntimeStatus.STOPPED

    def test_closed_desktop_session_requires_recalibration(self) -> None:
        """Invalidated authorized geometry ends tracking without pointer guesses."""
        camera = FakeCamera(deque(object() for _ in range(5)))
        vision = FakeVision(deque(observation(index) for index in range(5)))
        pointer = FakePointer((DisplayRegion("only", 0, 0, 100, 100),))
        status = FakeStatus()

        async def invalidate_after_calibration(_delay: float) -> None:
            if len(pointer.moves) == 5:
                pointer.closed = True
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                asyncio.Event(),
                tracking=fast_tracking(),
                sleep=invalidate_after_calibration,
            )
        )
        assert result == 3
        assert RuntimeStatus.RECALIBRATION_REQUIRED in [item[0] for item in status.reports]
        assert camera.closed
        assert vision.closed
        assert pointer.closed


if __name__ == "__main__":
    unittest.main()
