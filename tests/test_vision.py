"""Model-free tests for vision feature extraction."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gazeebo.vision import OpenSeeFaceEstimator, VisionError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(slots=True)
class StubFace:
    """Expose the OpenSeeFace fields consumed by the adapter."""

    conf: float
    eye_state: Sequence[Sequence[float]] | None
    euler: Sequence[float] | None
    lms: Sequence[Sequence[float]] | None


@dataclass(slots=True)
class StubTracker:
    """Return a fixed face list."""

    faces: Sequence[StubFace]

    def predict(self, frame: object) -> Sequence[StubFace]:
        """Return configured observations."""
        del frame
        return self.faces


def landmarks(
    *,
    right_eye_height: float = 3.0,
    left_eye_height: float = 3.0,
) -> list[list[float]]:
    """Create non-degenerate landmarks with independently shaped eyes."""
    points = [[10.0 + 100.0 * index / 65.0, 20.0 + 60.0 * index / 65.0, 1.0] for index in range(66)]

    def set_eye(start: int, center_x: float, height: float) -> None:
        points[start : start + 6] = [
            [center_x - 10.0, 50.0, 1.0],
            [center_x - 5.0, 50.0 - height, 1.0],
            [center_x + 5.0, 50.0 - height, 1.0],
            [center_x + 10.0, 50.0, 1.0],
            [center_x + 5.0, 50.0 + height, 1.0],
            [center_x - 5.0, 50.0 + height, 1.0],
        ]

    set_eye(36, 40.0, right_eye_height)
    set_eye(42, 80.0, left_eye_height)
    return points


def tracker_factory(faces: Sequence[StubFace]) -> Callable[[int, int], StubTracker]:
    """Bind a typed tracker result to the factory contract."""

    def create(_width: int, _height: int) -> StubTracker:
        return StubTracker(faces)

    return create


class VisionTests(unittest.TestCase):
    """Lock eye anatomy, confidence, and feature normalization."""

    def test_extracts_independent_eye_state_and_normalized_features(self) -> None:
        """The adapter maps tracker right/left order to anatomical fields."""
        face = StubFace(
            conf=0.92,
            eye_state=((0.2, 50.0, 40.0, 0.90), (0.8, 50.0, 80.0, 0.95)),
            euler=(0.1, -0.2, 0.0),
            lms=landmarks(right_eye_height=0.8),
        )
        tracker = StubTracker((face,))
        estimator = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=lambda _width, _height: tracker,
        )

        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        observation = estimator.observe(frame, 12.5)
        assert observation is not None
        assert observation.timestamp == 12.5
        assert observation.left_open > 0.8
        assert observation.right_open < 0.2
        assert observation.confidence == 0.90
        self.assertAlmostEqual(observation.features[0], 0.7)
        self.assertAlmostEqual(observation.features[1], 0.5)
        self.assertAlmostEqual(observation.features[2], 0.3)
        self.assertAlmostEqual(observation.features[3], 0.5)
        self.assertAlmostEqual(observation.features[6], 0.3)
        self.assertAlmostEqual(observation.features[7], 0.5)
        self.assertAlmostEqual(observation.features[8], 0.5)
        self.assertAlmostEqual(observation.features[9], 0.5)
        assert len(observation.context) == 9
        self.assertAlmostEqual(observation.context[3], 0.3)
        self.assertAlmostEqual(observation.context[4], 0.5)
        self.assertAlmostEqual(observation.context[7], 128.0 / 255.0)
        self.assertAlmostEqual(observation.context[8], 0.0)

    def test_rejects_missing_ambiguous_and_low_confidence_faces(self) -> None:
        """Tracking pauses rather than guessing at a face or weak eye result."""
        weak = StubFace(
            conf=0.9,
            eye_state=((1.0, 50.0, 40.0, 0.2), (1.0, 50.0, 80.0, 0.9)),
            euler=(0.0, 0.0),
            lms=landmarks(),
        )
        for faces in ((), (weak,), (weak, weak)):
            estimator = OpenSeeFaceEstimator(
                200,
                100,
                tracker_factory=tracker_factory(faces),
            )
            assert estimator.observe(object(), 1.0) is None

    def test_close_is_idempotent_and_blocks_future_inference(self) -> None:
        """Released model resources cannot process another frame."""
        estimator = OpenSeeFaceEstimator(
            200,
            100,
            tracker_factory=lambda _width, _height: StubTracker(()),
        )
        estimator.close()
        estimator.close()
        with self.assertRaisesRegex(VisionError, "closed"):
            estimator.observe(object(), 1.0)


if __name__ == "__main__":
    unittest.main()
