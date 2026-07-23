"""Tests for adaptive calibration targets, sampling, and validation."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import dataclass, field

from gazeebo.calibration import CalibrationModel, CalibrationSample
from gazeebo.contracts import DisplayRegion, EyeObservation, RuntimeStatus
from gazeebo.game import (
    GameConfig,
    GameError,
    TargetMeasurement,
    game_metrics,
    game_targets,
    run_adaptive_game,
)
from gazeebo.geometry import DisplayTopology, Point
from tests.fakes import FakePointer, FakeStatus


@dataclass(slots=True)
class FakeClock:
    """Advance deterministic monotonic time through injected sleeps."""

    value: float = 0.0

    def __call__(self) -> float:
        """Return current fixture time."""
        return self.value

    async def sleep(self, delay: float) -> None:
        """Advance fixture time without blocking."""
        self.value += delay
        await asyncio.sleep(0)


@dataclass(slots=True)
class FakeGameSurface:
    """Expose the visible target to a synthetic gaze estimator."""

    targets: list[tuple[str, float, float, float, str]] = field(default_factory=list)
    current: tuple[str, float, float] = ("", 0.0, 0.0)
    closed: bool = False

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Record and expose one current target."""
        self.current = (region_id, x, y)
        self.targets.append((region_id, x, y, diameter, label))

    async def close(self) -> None:
        """Record game-surface cleanup."""
        self.closed = True


@dataclass(slots=True)
class SyntheticCamera:
    """Return transient fixture frames indefinitely."""

    camera_id: str = "fixture-camera"

    def read(self) -> object:
        """Return one ephemeral frame marker."""
        return object()

    def close(self) -> None:
        """Satisfy the camera lifecycle contract."""


@dataclass(slots=True)
class TargetVision:
    """Produce stable features for the target currently shown."""

    surface: FakeGameSurface
    width: float
    height: float
    unstable: bool = False
    count: int = 0

    def observe(self, frame: object, timestamp: float) -> EyeObservation:
        """Derive synthetic features from the visible target."""
        del frame
        self.count += 1
        _, x, y = self.surface.current
        offset = 0.4 * (-1.0 if self.count % 2 else 1.0)
        if self.unstable:
            offset = 1000.0 * (-1.0 if self.count % 2 else 1.0)
        return EyeObservation(
            timestamp,
            1.0,
            1.0,
            (x / self.width + offset / self.width, y / self.height),
            0.9,
            (0.0, 0.5),
        )

    def close(self) -> None:
        """Satisfy the estimator lifecycle contract."""


def base_calibration() -> tuple[tuple[CalibrationSample, ...], CalibrationModel]:
    """Create a deliberately under-scaled initial model."""
    points = (Point(100.0, 100.0), Point(500.0, 350.0), Point(900.0, 600.0))
    samples = tuple(
        CalibrationSample((point.x / 2000.0, point.y / 1400.0), point) for point in points
    )
    return samples, CalibrationModel.fit(samples)


class GameTests(unittest.TestCase):
    """Lock deterministic game behavior without a graphical session."""

    def test_targets_are_varied_deterministic_and_cover_all_displays(self) -> None:
        """Unseen batches vary across regions without leaving any display."""
        regions = (
            DisplayRegion("left", 0, 0, 1000, 700),
            DisplayRegion("right", 1000, 200, 800, 600),
        )
        first = game_targets(regions, 10, 40.0, 120.0)
        repeated = game_targets(regions, 10, 40.0, 120.0)
        second = game_targets(regions, 5, 40.0, 120.0, start_index=10)
        assert first == repeated
        assert {target.region_id for target in first} == {"left", "right"}
        assert {(target.region_id, target.x, target.y) for target in first}.isdisjoint(
            (target.region_id, target.x, target.y) for target in second
        )
        assert len({target.diameter for target in first}) > 1
        assert any(target.edge_or_corner for target in first)
        assert any(not target.edge_or_corner for target in first)
        by_id = {region.region_id: region for region in regions}
        for target in first:
            region = by_id[target.region_id]
            radius = target.diameter / 2.0
            assert radius <= target.x <= region.width - radius
            assert radius <= target.y <= region.height - radius

    def test_iterative_defaults_and_terminal_limits_are_finite(self) -> None:
        """Default batches stop at precision or after exactly 55 circles."""
        config = GameConfig()
        assert config.batch_size == 5
        assert config.precision_threshold == 100.0
        assert config.maximum_targets == 55
        with self.assertRaisesRegex(ValueError, "multiple"):
            GameConfig(batch_size=5, maximum_targets=54)

    def test_metrics_keep_edge_and_response_measurements_separate(self) -> None:
        """Aggregate reporting preserves holdout error categories."""
        metrics = game_metrics(
            (
                TargetMeasurement(10.0, False, 0.4),
                TargetMeasurement(30.0, True, None),
                TargetMeasurement(20.0, True, 0.2),
            )
        )
        assert metrics.target_count == 3
        assert metrics.hit_count == 2
        assert metrics.median_error == 20.0
        assert metrics.edge_error == 25.0
        self.assertAlmostEqual(metrics.median_response or 0.0, 0.3)

    def test_adaptive_training_improves_independent_validation(self) -> None:
        """Stable game samples improve a deliberately biased model."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        pointer = FakePointer((region,))
        surface = FakeGameSurface()
        vision = TargetVision(surface, region.width, region.height)
        status = FakeStatus()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        config = GameConfig(
            batch_size=5,
            maximum_targets=15,
            precision_threshold=250.0,
            settle_seconds=0.0,
            dwell_seconds=0.03,
            target_timeout_seconds=0.50,
            minimum_diameter=40.0,
            maximum_diameter=80.0,
            stability_threshold=0.05,
        )
        result = asyncio.run(
            run_adaptive_game(
                SyntheticCamera(),
                vision,
                pointer,
                topology,
                surface,
                status,
                asyncio.Event(),
                samples,
                initial_model,
                config,
                closed_threshold=0.35,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert result.after.median_error < result.before.median_error
        assert result.precision_met
        assert len(surface.targets) in (10, 15)
        assert all(move[0] == "selected" for move in pointer.moves)
        states = [report[0] for report in status.reports]
        assert states.count(RuntimeStatus.GAME_VALIDATING) >= 4
        assert RuntimeStatus.GAME_TRAINING in states

    def test_explicit_training_adds_one_reported_batch_before_final_holdout(self) -> None:
        """On-demand training augments a passing model and retains a fresh holdout."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeGameSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        result = asyncio.run(
            run_adaptive_game(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height),
                FakePointer((region,)),
                topology,
                surface,
                FakeStatus(),
                asyncio.Event(),
                samples,
                initial_model,
                GameConfig(
                    batch_size=5,
                    maximum_targets=10,
                    precision_threshold=10000.0,
                    settle_seconds=0.0,
                    dwell_seconds=0.03,
                    target_timeout_seconds=0.50,
                    minimum_diameter=40.0,
                    maximum_diameter=80.0,
                    stability_threshold=0.05,
                ),
                force_adaptation=True,
                closed_threshold=0.35,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert result.precision_met
        assert len(result.accepted_targets) == 5
        assert len(surface.targets) == 10

    def test_maximum_failure_keeps_terminal_batch_unseen(self) -> None:
        """The last reported batch is not added after precision failure."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeGameSurface()
        status = FakeStatus()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        config = GameConfig(
            batch_size=5,
            maximum_targets=10,
            precision_threshold=1e-9,
            settle_seconds=0.0,
            dwell_seconds=0.03,
            target_timeout_seconds=0.50,
            minimum_diameter=40.0,
            maximum_diameter=80.0,
            stability_threshold=0.05,
        )
        result = asyncio.run(
            run_adaptive_game(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height),
                FakePointer((region,)),
                topology,
                surface,
                status,
                asyncio.Event(),
                samples,
                initial_model,
                config,
                closed_threshold=0.35,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert not result.precision_met
        assert len(result.accepted_targets) == 5
        assert len(result.holdout_targets) == 5
        assert len(surface.targets) == 10
        states = [item[0] for item in status.reports]
        assert states.count(RuntimeStatus.GAME_TRAINING) == 2
        assert RuntimeStatus.TRAINING_RECOMMENDED in states
        assert "retaining batch" in status.reports[-1][1]

    def test_prior_initial_targets_count_toward_invocation_maximum(self) -> None:
        """Initial anchors leave only complete unseen batches inside the 55-target cap."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeGameSurface()
        status = FakeStatus()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        result = asyncio.run(
            run_adaptive_game(
                SyntheticCamera(),
                TargetVision(surface, region.width, region.height),
                FakePointer((region,)),
                topology,
                surface,
                status,
                asyncio.Event(),
                samples,
                initial_model,
                GameConfig(
                    batch_size=5,
                    maximum_targets=10,
                    precision_threshold=1e-9,
                    settle_seconds=0.0,
                    dwell_seconds=0.03,
                    target_timeout_seconds=0.50,
                    minimum_diameter=40.0,
                    maximum_diameter=80.0,
                    stability_threshold=0.05,
                ),
                target_offset=5,
                closed_threshold=0.35,
                pointer_interval=0.0,
                frame_interval=0.01,
                clock=clock,
                sleep=clock.sleep,
            )
        )
        assert result is not None
        assert not result.precision_met
        assert len(surface.targets) == 5
        assert any("total 10/10" in detail for _state, detail in status.reports)

    def test_unstable_samples_fail_in_finite_time_without_refitting(self) -> None:
        """Unstable gaze cannot train a model or loop forever."""
        region = DisplayRegion("selected", 0, 0, 1000, 700)
        topology = DisplayTopology((region,))
        surface = FakeGameSurface()
        clock = FakeClock()
        samples, initial_model = base_calibration()
        config = GameConfig(
            batch_size=5,
            maximum_targets=10,
            precision_threshold=100.0,
            settle_seconds=0.0,
            dwell_seconds=0.03,
            target_timeout_seconds=0.08,
            minimum_diameter=40.0,
            maximum_diameter=80.0,
            stability_threshold=0.01,
        )
        with self.assertRaisesRegex(GameError, "stable open-eye"):
            asyncio.run(
                run_adaptive_game(
                    SyntheticCamera(),
                    TargetVision(surface, region.width, region.height, unstable=True),
                    FakePointer((region,)),
                    topology,
                    surface,
                    FakeStatus(),
                    asyncio.Event(),
                    samples,
                    initial_model,
                    config,
                    closed_threshold=0.35,
                    pointer_interval=0.1,
                    frame_interval=0.01,
                    clock=clock,
                    sleep=clock.sleep,
                )
            )
