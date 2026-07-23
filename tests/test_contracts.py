"""Tests for hardware-independent runtime boundaries."""

from __future__ import annotations

import unittest
from collections import deque

from gazeebo.contracts import DisplayRegion, EyeObservation, RuntimeStatus
from tests.fakes import FakeCamera, FakePointer, FakeStatus, FakeVision


class ContractTests(unittest.TestCase):
    """Lock validation and fake-resource behavior."""

    def test_region_requires_positive_dimensions(self) -> None:
        """Invalid authorized geometry fails before calibration."""
        with self.assertRaisesRegex(ValueError, "dimensions"):
            DisplayRegion("bad", 0, 0, 0, 10)

    def test_observation_requires_normalized_confidence(self) -> None:
        """Vision adapters cannot leak unnormalized confidence into policy."""
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            EyeObservation(1.0, 1.2, 0.5, (0.0,), 0.9, (0.0,))

    def test_camera_and_vision_fakes_close_idempotently(self) -> None:
        """Camera-free fixtures enforce the process cleanup contract."""
        observation = EyeObservation(
            1.0,
            0.9,
            0.8,
            (0.1, 0.2),
            0.95,
            (0.0, 0.5),
        )
        camera = FakeCamera(deque([object()]))
        vision = FakeVision(deque([observation]))

        frame = camera.read()
        assert vision.observe(frame, 1.0) == observation
        camera.close()
        camera.close()
        vision.close()
        vision.close()
        assert camera.closed
        assert vision.closed

    def test_pointer_fake_rejects_out_of_region_motion(self) -> None:
        """Policy tests cannot silently target unauthorized coordinates."""
        region = DisplayRegion("display-a", -100, 50, 800, 600)
        pointer = FakePointer((region,))
        pointer.move("display-a", 799.0, 599.0)

        assert pointer.moves == [("display-a", 799.0, 599.0)]
        with self.assertRaisesRegex(ValueError, "outside"):
            pointer.move("display-a", 800.0, 10.0)

    def test_status_fake_records_ordered_transitions(self) -> None:
        """Runtime state reporting remains observable in deterministic tests."""
        status = FakeStatus()
        status.report(RuntimeStatus.STARTING)
        status.report(RuntimeStatus.CAMERA_ERROR, "capture unavailable")
        status.report(RuntimeStatus.STOPPED)

        assert status.reports == [
            (RuntimeStatus.STARTING, ""),
            (RuntimeStatus.CAMERA_ERROR, "capture unavailable"),
            (RuntimeStatus.STOPPED, ""),
        ]


if __name__ == "__main__":
    unittest.main()
