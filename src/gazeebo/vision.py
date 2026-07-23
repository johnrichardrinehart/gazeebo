"""In-process face, pupil, and eye-state estimation."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from gazeebo.contracts import EyeObservation, Frame

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


LANDMARK_COUNT = 66
LANDMARK_DIMENSIONS = 2
EYE_STATE_SHAPE = (2, 4)
RIGHT_EYE_INDICES = (36, 37, 38, 39, 40, 41)
LEFT_EYE_INDICES = (42, 43, 44, 45, 46, 47)
CLOSED_EYE_RATIO = 0.08
EYE_RATIO_RANGE = 0.18
GAZE_OPEN_THRESHOLD = 0.35
MINIMUM_EYE_WIDTH = 1e-6
ROLL_INDEX = 2
MINIMUM_IMAGE_DIMENSIONS = 2
COLOR_IMAGE_DIMENSIONS = 3


class VisionError(RuntimeError):
    """The local vision model could not produce a safe observation."""


class _Face(Protocol):
    conf: float
    eye_state: Sequence[Sequence[float]] | None
    euler: Sequence[float] | None
    lms: Sequence[Sequence[float]] | None


class _Tracker(Protocol):
    def predict(self, frame: Frame) -> Sequence[_Face]:
        """Return tracked faces."""


def _default_tracker_factory(width: int, height: int) -> _Tracker:
    """Load the packaged OpenSeeFace tracker without starting its UDP executable."""
    tracker_directory = os.environ.get("GAZEEBO_TRACKER_DIR")
    if tracker_directory and tracker_directory not in sys.path:
        sys.path.insert(0, tracker_directory)
    try:
        from tracker import Tracker  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as error:
        msg = "packaged face tracker is unavailable"
        raise VisionError(msg) from error
    model_directory = os.environ.get("GAZEEBO_MODEL_DIR")
    return cast(
        "_Tracker",
        Tracker(
            width,
            height,
            model_type=1,
            max_faces=1,
            max_threads=4,
            silent=True,
            model_dir=model_directory,
            feature_level=2,
        ),
    )


def _eye_openness(points: np.ndarray, indices: tuple[int, ...]) -> float:
    """Normalize one six-landmark eye aspect ratio to an open confidence."""
    eye = points[np.asarray(indices)]
    horizontal = float(np.linalg.norm(eye[0] - eye[3]))
    if horizontal <= MINIMUM_EYE_WIDTH:
        return 0.0
    vertical = float(np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4]))
    ratio = vertical / (2.0 * horizontal)
    return float(np.clip((ratio - CLOSED_EYE_RATIO) / EYE_RATIO_RANGE, 0.0, 1.0))


class OpenSeeFaceEstimator:
    """Convert OpenSeeFace output into ephemeral normalized gaze features."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        minimum_confidence: float = 0.55,
        tracker_factory: Callable[[int, int], _Tracker] | None = None,
    ) -> None:
        """Create one in-process tracker for fixed camera dimensions."""
        if width <= 0 or height <= 0:
            msg = "vision dimensions must be positive"
            raise ValueError(msg)
        if not 0.0 <= minimum_confidence <= 1.0:
            msg = "minimum vision confidence must be between zero and one"
            raise ValueError(msg)
        self._width = width
        self._height = height
        self._minimum_confidence = minimum_confidence
        self._tracker: _Tracker | None = (tracker_factory or _default_tracker_factory)(
            width,
            height,
        )

    def observe(  # noqa: PLR0911
        self,
        frame: Frame,
        timestamp: float,
    ) -> EyeObservation | None:
        """Return the only confident face, independent eye state, and gaze features."""
        if self._tracker is None:
            msg = "vision estimator is closed"
            raise VisionError(msg)
        faces = self._tracker.predict(frame)
        if len(faces) != 1:
            return None
        face = faces[0]
        if face.eye_state is None or face.lms is None:
            return None
        eyes = np.asarray(face.eye_state, dtype=np.float64)
        landmarks = np.asarray(face.lms, dtype=np.float64)
        if (
            eyes.shape != EYE_STATE_SHAPE
            or landmarks.ndim != LANDMARK_DIMENSIONS
            or landmarks.shape[0] < LANDMARK_COUNT
        ):
            return None

        face_points = landmarks[:LANDMARK_COUNT, :LANDMARK_DIMENSIONS]
        minimum = face_points.min(axis=0)
        maximum = face_points.max(axis=0)
        size = maximum - minimum
        if np.any(size <= 1.0):
            return None

        _, right_y, right_x, right_confidence = eyes[0]
        _, left_y, left_x, left_confidence = eyes[1]
        right_open = _eye_openness(face_points, RIGHT_EYE_INDICES)
        left_open = _eye_openness(face_points, LEFT_EYE_INDICES)
        face_confidence = float(face.conf)
        pupil_confidence = min(float(right_confidence), float(left_confidence))
        if not np.isfinite(face_confidence) or face_confidence < self._minimum_confidence:
            return None
        if not np.isfinite(pupil_confidence):
            return None
        if (
            left_open > GAZE_OPEN_THRESHOLD
            and right_open > GAZE_OPEN_THRESHOLD
            and pupil_confidence < self._minimum_confidence
        ):
            return None
        confidence = float(np.clip(min(face_confidence, pupil_confidence), 0.0, 1.0))

        euler = np.asarray(
            face.euler if face.euler is not None else (0.0, 0.0, 0.0),
            dtype=np.float64,
        )
        pitch = float(euler[0]) if euler.size > 0 and np.isfinite(euler[0]) else 0.0
        yaw = float(euler[1]) if euler.size > 1 and np.isfinite(euler[1]) else 0.0
        roll = (
            float(euler[ROLL_INDEX])
            if euler.size > ROLL_INDEX and np.isfinite(euler[ROLL_INDEX])
            else 0.0
        )
        center = (minimum + maximum) / 2.0
        normalized_left_x = float((left_x - minimum[0]) / size[0])
        normalized_left_y = float((left_y - minimum[1]) / size[1])
        normalized_right_x = float((right_x - minimum[0]) / size[0])
        normalized_right_y = float((right_y - minimum[1]) / size[1])
        features = (
            normalized_left_x,
            normalized_left_y,
            normalized_right_x,
            normalized_right_y,
            pitch / 180.0,
            yaw / 180.0,
            float(center[0] / self._width),
            float(center[1] / self._height),
            (normalized_left_x + normalized_right_x) / 2.0,
            (normalized_left_y + normalized_right_y) / 2.0,
        )
        luminance_mean, luminance_spread = _illumination(frame)
        context = (
            pitch / 180.0,
            yaw / 180.0,
            roll / 180.0,
            float(center[0] / self._width),
            float(center[1] / self._height),
            float(size[0] / self._width),
            float(size[1] / self._height),
            luminance_mean,
            luminance_spread,
        )
        if not all(np.isfinite(value) for value in (*features, *context)):
            return None
        return EyeObservation(
            timestamp=timestamp,
            left_open=float(np.clip(left_open, 0.0, 1.0)),
            right_open=float(np.clip(right_open, 0.0, 1.0)),
            features=features,
            confidence=confidence,
            context=context,
        )

    def close(self) -> None:
        """Release model references idempotently."""
        self._tracker = None


def _illumination(frame: Frame) -> tuple[float, float]:
    """Reduce a frame to bounded luminance context without retaining pixels."""
    try:
        pixels = np.asarray(frame, dtype=np.float64)
    except (TypeError, ValueError):
        return 0.5, 0.0
    if pixels.ndim < MINIMUM_IMAGE_DIMENSIONS or pixels.size == 0:
        return 0.5, 0.0
    if pixels.ndim >= COLOR_IMAGE_DIMENSIONS:
        pixels = pixels[..., :3].mean(axis=-1)
    mean = float(np.clip(pixels.mean() / 255.0, 0.0, 1.0))
    spread = float(np.clip(pixels.std() / 128.0, 0.0, 1.0))
    return mean, spread
