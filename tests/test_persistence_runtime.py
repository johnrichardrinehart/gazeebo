"""Tests for persisted model reuse in the foreground runtime."""

from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from collections import deque
from pathlib import Path

from gazeebo.calibration import CalibrationModel, CalibrationSample
from gazeebo.contexts import build_router
from gazeebo.contracts import DisplayRegion, EyeObservation, RuntimeStatus
from gazeebo.game import CollectedTarget, GameConfig
from gazeebo.geometry import DisplayTopology, Point, calibration_targets
from gazeebo.runtime import TrackingConfig, _persist_targets, run_owned_session
from gazeebo.state import TrainingState, TrainingStore
from tests.fakes import FakeCamera, FakeGame, FakePointer, FakeStatus, FakeVision


def collected(topology: DisplayTopology) -> tuple[CollectedTarget, ...]:
    """Create a complete one-display training set with normalized context."""
    return tuple(
        CollectedTarget(
            (
                topology.to_global(target).x,
                topology.to_global(target).y,
            ),
            (0.0, 0.5),
            target,
            "center" if index == 2 else "corner",
        )
        for index, target in enumerate(calibration_targets(topology))
    )


class _ContextIncumbent:
    """Decode test holdout coordinates supplied only to the incumbent."""

    kind = "fixture"

    def predict(
        self,
        _features: tuple[float, ...],
        context: tuple[float, ...] | None = None,
    ) -> Point:
        assert context is not None
        return Point(context[0] * 1000.0, context[1] * 700.0)


def fast_reuse() -> TrackingConfig:
    """Use one passive context and one tracking observation."""
    return TrackingConfig(
        calibration_settle_seconds=0.0,
        calibration_samples_per_target=1,
        calibration_attempts_per_target=1,
        startup_context_samples=1,
        startup_context_attempts=1,
        frame_interval_seconds=0.0,
        maximum_missed_frames=2,
    )


class _Clock:
    """Advance deterministic adaptive-game time through awaited sleeps."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    async def sleep(self, delay: float) -> None:
        self.value += max(delay, 0.01)
        await asyncio.sleep(0)


class _TargetVision:
    """Observe the active fake target with one stable routing context."""

    def __init__(self, game: FakeGame) -> None:
        self.game = game
        self.closed = False

    def observe(self, _frame: object, timestamp: float) -> EyeObservation:
        if self.game.targets:
            _region, x, y, _diameter, _label = self.game.targets[-1]
        else:
            x, y = 500.0, 350.0
        return EyeObservation(timestamp, 1.0, 1.0, (x, y), 1.0, (0.0, 0.5))

    def close(self) -> None:
        self.closed = True


class _EndlessCamera:
    """Supply disposable frames until the runtime closes it."""

    camera_id = "fixture-camera"

    def __init__(self) -> None:
        self.closed = False

    def read(self) -> object:
        return object()

    def close(self) -> None:
        self.closed = True


class PersistenceRuntimeTests(unittest.TestCase):
    """Lock atomic candidate commits and calibration-free repeated startup."""

    def test_accepted_targets_commit_models_and_reload(self) -> None:
        """A training transaction persists aggregates, coefficients, and quality."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            persisted = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v1",
                collected(topology),
                80.0,
                90.0,
            )
            assert persisted is not None
            router, candidate = persisted
            loaded = store.load()
            assert loaded == candidate
            assert len(loaded.targets) == 5
            assert loaded.clusters
            assert loaded.models
            assert loaded.validations[-1].median_error == 80.0
            point = router.predict((500.0, 350.0), (0.0, 0.5))
            assert 0.0 <= point.x < 1000.0
            assert 0.0 <= point.y < 700.0

    def test_exact_restart_loads_the_terminally_validated_coefficients(self) -> None:
        """Exact topology reuse does not replace measured coefficients by refitting."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        targets = collected(topology)
        validated = CalibrationModel.fit(
            [
                CalibrationSample(
                    target.features,
                    Point(
                        topology.to_global(target.target).x + 40.0,
                        topology.to_global(target.target).y + 20.0,
                    ),
                )
                for target in targets
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            persisted = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v1",
                targets,
                80.0,
                90.0,
                validated_model=validated,
            )
            assert persisted is not None
            before = persisted[0].predict((500.0, 350.0), (0.0, 0.5))
            after = build_router(
                store.load(),
                topology,
                camera_id="fixture-camera",
                feature_schema="gaze-v1",
            ).predict((500.0, 350.0), (0.0, 0.5))
            assert math.isclose(after.x, before.x)
            assert math.isclose(after.y, before.y)
            assert after.x > 500.0
            assert after.y > 350.0

    def test_failed_persistent_router_holdout_leaves_store_unchanged(self) -> None:
        """A refitted context model cannot save on the game model's proxy result."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        bad_holdout = tuple(
            CollectedTarget(
                (0.0, 0.0),
                (target.x / 1000.0, target.y / 700.0),
                target,
                "corner",
            )
            for target in calibration_targets(topology)[:3]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gazeebo" / "training-v1.json"
            result = _persist_targets(
                TrainingStore(path),
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v1",
                collected(topology),
                0.0,
                0.0,
                holdout=bad_holdout,
                incumbent=_ContextIncumbent(),
            )
            assert result is None
            assert not path.exists()

    def test_finite_terminal_result_can_establish_first_model(self) -> None:
        """A first below-threshold run remains available for later improvement."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        holdout = tuple(
            CollectedTarget((0.0, 0.0), (0.0, 0.5), target, "corner")
            for target in calibration_targets(topology)[:3]
        )
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            result = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v1",
                collected(topology),
                0.0,
                0.0,
                holdout=holdout,
            )
            assert result is not None
            assert store.load().validations[-1].median_error > 100.0

    def test_repeated_run_uses_passive_context_without_calibration_targets(self) -> None:
        """A compatible store starts navigation after passive model routing."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            persisted = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v1",
                collected(topology),
                80.0,
                90.0,
            )
            assert persisted is not None
            _router, state = persisted
            observations: deque[EyeObservation | None] = deque(
                (
                    EyeObservation(0.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
                    EyeObservation(1.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
                )
            )
            camera = FakeCamera(deque((object(), object())))
            pointer = FakePointer(topology.regions)
            status = FakeStatus()
            stop = asyncio.Event()

            async def stop_after_move(_delay: float) -> None:
                if pointer.moves and status.reports[-1][0] is RuntimeStatus.ACTIVE:
                    stop.set()
                await asyncio.sleep(0)

            result = asyncio.run(
                run_owned_session(
                    camera,
                    FakeVision(observations),
                    pointer,
                    status,
                    stop,
                    tracking=fast_reuse(),
                    training_state=state,
                    sleep=stop_after_move,
                )
            )
            assert result == 0
            states = [item[0] for item in status.reports]
            assert RuntimeStatus.SELECTING_MODEL in states
            assert RuntimeStatus.INITIAL_TRAINING not in states
            assert pointer.moves
            assert camera.closed
            assert pointer.closed

    def test_active_request_transitions_navigation_into_training_and_back(self) -> None:
        """An active event runs on-demand targets without replacing the process."""
        topology = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            persisted = _persist_targets(
                store,
                TrainingState(),
                topology,
                "fixture-camera",
                "gaze-v1",
                collected(topology),
                80.0,
                90.0,
            )
            assert persisted is not None
            _router, state = persisted
        game = FakeGame()
        camera = _EndlessCamera()
        vision = _TargetVision(game)
        pointer = FakePointer(topology.regions)
        status = FakeStatus()
        stop = asyncio.Event()
        request = asyncio.Event()
        request.set()
        clock = _Clock()

        async def stop_after_second_active(delay: float) -> None:
            await clock.sleep(delay)
            active_count = sum(item[0] is RuntimeStatus.ACTIVE for item in status.reports)
            if active_count >= 2 and pointer.moves:
                stop.set()

        result = asyncio.run(
            run_owned_session(
                camera,
                vision,
                pointer,
                status,
                stop,
                tracking=fast_reuse(),
                game_config=GameConfig(
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
                training_state=state,
                training_requested_event=request,
                game_factory=lambda _regions: game,
                clock=clock,
                sleep=stop_after_second_active,
            )
        )
        assert result == 0
        assert len(game.targets) == 10
        assert sum(item[0] is RuntimeStatus.ACTIVE for item in status.reports) == 2
        assert not request.is_set()
        assert game.closed
        assert camera.closed
        assert vision.closed
        assert pointer.closed

    def test_added_output_uses_weak_best_effort_model_without_forced_training(self) -> None:
        """A changed topology remains usable but is explicitly unvalidated."""
        source = DisplayTopology((DisplayRegion("stable", 0, 0, 1000, 700),))
        with tempfile.TemporaryDirectory() as temporary:
            store = TrainingStore(Path(temporary) / "gazeebo" / "training-v1.json")
            persisted = _persist_targets(
                store,
                TrainingState(),
                source,
                "fixture-camera",
                "gaze-v1",
                collected(source),
                80.0,
                90.0,
            )
            assert persisted is not None
            _router, state = persisted
        current = DisplayTopology(
            (
                DisplayRegion("stable", 0, 0, 1000, 700),
                DisplayRegion("added", 1200, 0, 800, 600),
            )
        )
        observations: deque[EyeObservation | None] = deque(
            (
                EyeObservation(0.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
                EyeObservation(1.0, 1.0, 1.0, (500.0, 350.0), 1.0, (0.0, 0.5)),
            )
        )
        pointer = FakePointer(current.regions)
        status = FakeStatus()
        stop = asyncio.Event()

        async def stop_after_move(_delay: float) -> None:
            if pointer.moves and status.reports[-1][0] is RuntimeStatus.ACTIVE:
                stop.set()
            await asyncio.sleep(0)

        result = asyncio.run(
            run_owned_session(
                FakeCamera(deque((object(), object()))),
                FakeVision(observations),
                pointer,
                status,
                stop,
                tracking=fast_reuse(),
                training_state=state,
                sleep=stop_after_move,
            )
        )
        assert result == 0
        states = [item[0] for item in status.reports]
        assert RuntimeStatus.TOPOLOGY_UNVALIDATED in states
        assert RuntimeStatus.INITIAL_TRAINING not in states
        assert all(
            any(
                region.region_id == region_id and 0 <= x < region.width and 0 <= y < region.height
                for region in current.regions
            )
            for region_id, x, y in pointer.moves
        )


if __name__ == "__main__":
    unittest.main()
