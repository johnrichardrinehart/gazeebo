"""Hardware-independent runtime contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

type Frame = object
type FeatureVector = tuple[float, ...]
type ContextVector = tuple[float, ...]


class RuntimeStatus(Enum):
    """Observable process states."""

    STARTING = "starting"
    LOADING = "loading"
    AUTHORIZING = "authorizing"
    SELECTING_MODEL = "selecting-model"
    TOPOLOGY_UNVALIDATED = "topology-unvalidated"
    INITIAL_TRAINING = "initial-training"
    GAME_VALIDATING = "validating"
    GAME_TRAINING = "adaptive-training"
    TRAINING_RECOMMENDED = "training-recommended"
    GAME_ERROR = "game-error"
    MODEL_ERROR = "model-error"
    STORE_ERROR = "store-error"
    ACTIVE = "active"
    RECALIBRATION_REQUIRED = "recalibration-required"
    CAMERA_ERROR = "camera-error"
    INPUT_ERROR = "input-error"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class DisplayRegion:
    """One pointer region authorized for the current session."""

    region_id: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject geometry that cannot receive a pointer coordinate."""
        if not self.region_id:
            msg = "display region ID must not be empty"
            raise ValueError(msg)
        if self.width <= 0 or self.height <= 0:
            msg = "display region dimensions must be positive"
            raise ValueError(msg)

    @property
    def right(self) -> int:
        """Return the exclusive right edge."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom edge."""
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class EyeObservation:
    """One timestamped vision result without retaining its source frame."""

    timestamp: float
    left_open: float
    right_open: float
    features: FeatureVector
    confidence: float
    context: ContextVector

    def __post_init__(self) -> None:
        """Validate normalized confidence values and a usable feature vector."""
        values = (self.left_open, self.right_open, self.confidence)
        if any(value < 0.0 or value > 1.0 for value in values):
            msg = "eye and face confidence values must be between zero and one"
            raise ValueError(msg)
        if not self.features or not self.context:
            msg = "gaze feature and routing context vectors must not be empty"
            raise ValueError(msg)


class CameraCapture(Protocol):
    """Own a local camera for exactly one process session."""

    @property
    def camera_id(self) -> str:
        """Return an opaque local fingerprint without exposing a device path."""

    def read(self) -> Frame:
        """Return the next in-memory frame or raise a camera error."""

    def close(self) -> None:
        """Release the camera; repeated calls must be safe."""


class VisionEstimator(Protocol):
    """Estimate gaze and independent eye state from an in-memory frame."""

    def observe(self, frame: Frame, timestamp: float) -> EyeObservation | None:
        """Return one confident face observation, or none when unavailable."""

    def close(self) -> None:
        """Release inference resources; repeated calls must be safe."""


class PointerController(Protocol):
    """Control the display regions authorized for this session."""

    @property
    def regions(self) -> tuple[DisplayRegion, ...]:
        """Return every authorized display region."""

    @property
    def topology_id(self) -> str:
        """Return an opaque identity that changes with selected geometry."""

    def move(self, region_id: str, x: float, y: float) -> None:
        """Move to a region-local logical coordinate."""

    async def close(self) -> None:
        """Drain motion events and close authorization idempotently."""


class DebugHud(Protocol):
    """Display rate-limited pointer diagnostics when explicitly enabled."""

    def set_model_context(self, routing: str, topology_quality: str) -> None:
        """Set safe routing and topology labels without exposing feature values."""

    async def update(self, region_id: str, x: float, y: float) -> None:
        """Show the current authorized region and global logical coordinate."""

    async def close(self) -> None:
        """Remove the HUD and release its desktop connection idempotently."""


class CalibrationGameSurface(Protocol):
    """Show transient click-through targets across authorized displays."""

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Replace the visible target on one authorized region."""

    async def close(self) -> None:
        """Remove the game surface and release its desktop connection."""


class StatusSink(Protocol):
    """Report process state without coupling policy to a user interface."""

    def report(self, status: RuntimeStatus, detail: str = "") -> None:
        """Publish a status transition and optional safe detail."""
