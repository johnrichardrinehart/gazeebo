"""Deterministic in-memory implementations of runtime boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import deque

    from gazeebo.contracts import DisplayRegion, EyeObservation, Frame, RuntimeStatus


@dataclass(slots=True)
class FakeCamera:
    """Yield a fixed frame sequence and record cleanup."""

    frames: deque[Frame]
    camera_id: str = "fixture-camera"
    closed: bool = False

    def read(self) -> Frame:
        """Return the next configured frame."""
        if self.closed:
            msg = "camera is closed"
            raise RuntimeError(msg)
        if not self.frames:
            msg = "camera fixture is exhausted"
            raise EOFError(msg)
        return self.frames.popleft()

    def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakeVision:
    """Yield configured observations without inspecting fixture frames."""

    observations: deque[EyeObservation | None]
    closed: bool = False

    def observe(self, frame: Frame, timestamp: float) -> EyeObservation | None:
        """Return the next configured observation."""
        del frame, timestamp
        if self.closed:
            msg = "vision estimator is closed"
            raise RuntimeError(msg)
        return self.observations.popleft()

    def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakePointer:
    """Record pointer events within fixed authorized geometry."""

    regions: tuple[DisplayRegion, ...]
    topology_id: str = "fixture-layout"
    moves: list[tuple[str, float, float]] = field(default_factory=list)
    closed: bool = False

    def move(self, region_id: str, x: float, y: float) -> None:
        """Record a validated region-local movement."""
        region = next((item for item in self.regions if item.region_id == region_id), None)
        if region is None:
            msg = f"unknown fixture region: {region_id}"
            raise ValueError(msg)
        if not 0 <= x < region.width or not 0 <= y < region.height:
            msg = "fixture movement is outside its region"
            raise ValueError(msg)
        self.moves.append((region_id, x, y))

    async def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakeHud:
    """Record opt-in pointer diagnostics and cleanup."""

    updates: list[tuple[str, float, float]] = field(default_factory=list)
    model_context: tuple[str, str] = ("unselected", "unknown")
    closed: bool = False

    def set_model_context(self, routing: str, topology_quality: str) -> None:
        """Record safe routing and topology labels."""
        self.model_context = (routing, topology_quality)

    async def update(self, region_id: str, x: float, y: float) -> None:
        """Record one global pointer diagnostic."""
        self.updates.append((region_id, x, y))

    async def close(self) -> None:
        """Record idempotent closure."""
        self.closed = True


@dataclass(slots=True)
class FakeGame:
    """Record transient calibration targets and cleanup."""

    targets: list[tuple[str, float, float, float, str]] = field(default_factory=list)
    closed: bool = False

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Record one target without opening a surface."""
        self.targets.append((region_id, x, y, diameter, label))

    async def close(self) -> None:
        """Record idempotent surface cleanup."""
        self.closed = True


@dataclass(slots=True)
class FakeStatus:
    """Collect status transitions."""

    reports: list[tuple[RuntimeStatus, str]] = field(default_factory=list)

    def report(self, status: RuntimeStatus, detail: str = "") -> None:
        """Record a transition."""
        self.reports.append((status, detail))
