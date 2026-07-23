"""Foreground model selection, training, navigation, and cleanup lifecycle."""

from __future__ import annotations

import asyncio
import copy
import math
import signal
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gazeebo.adaptation import TopologyQuality, make_stored_target
from gazeebo.calibration import CalibrationModel, CalibrationSample, aggregate_features
from gazeebo.contexts import (
    ModelRouter,
    ValidationMetrics,
    add_target,
    build_router,
    calibration_samples_for,
    candidate_is_acceptable,
)
from gazeebo.contracts import RuntimeStatus
from gazeebo.game import CollectedTarget, GameConfig, GazePredictor, run_adaptive_game
from gazeebo.geometry import DisplayTopology, PointerSmoother, PointerTarget, calibration_targets
from gazeebo.state import TrainingState, TrainingStore, ValidationSummary

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from gazeebo.contracts import (
        CalibrationGameSurface,
        CameraCapture,
        ContextVector,
        DebugHud,
        DisplayRegion,
        FeatureVector,
        PointerController,
        StatusSink,
        VisionEstimator,
    )
    from gazeebo.control import TrainingControl

FEATURE_SCHEMA = "gaze-v1"
TRAINING_REQUESTED_RESULT = 4
CALIBRATION_EDGE_BOUNDARY = 0.25


class TrackingError(RuntimeError):
    """Model selection, training, or navigation cannot continue safely."""


@dataclass(frozen=True, slots=True)
class _BaseCalibration:
    model: CalibrationModel
    samples: tuple[CalibrationSample, ...]
    targets: tuple[CollectedTarget, ...]


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Session timing and failure thresholds."""

    calibration_settle_seconds: float = 1.00
    calibration_samples_per_target: int = 8
    calibration_attempts_per_target: int = 80
    startup_context_samples: int = 8
    startup_context_attempts: int = 80
    frame_interval_seconds: float = 0.01
    maximum_missed_frames: int = 900
    smoothing_alpha: float = 0.35
    smoothing_dead_zone: float = 6.0
    smoothing_maximum_step: float = 600.0
    pointer_update_interval_seconds: float = 0.10
    open_eye_threshold: float = 0.35

    def __post_init__(self) -> None:
        """Reject unsafe or non-terminating runtime settings."""
        intervals = (
            self.calibration_settle_seconds,
            self.frame_interval_seconds,
            self.pointer_update_interval_seconds,
        )
        if any(interval < 0.0 for interval in intervals):
            msg = "tracking intervals must be non-negative"
            raise ValueError(msg)
        counts = (
            self.calibration_samples_per_target,
            self.calibration_attempts_per_target,
            self.startup_context_samples,
            self.startup_context_attempts,
            self.maximum_missed_frames,
        )
        if any(count <= 0 for count in counts):
            msg = "tracking sample and failure counts must be positive"
            raise ValueError(msg)
        if self.calibration_attempts_per_target < self.calibration_samples_per_target:
            msg = "calibration attempts must cover the requested samples"
            raise ValueError(msg)
        if self.startup_context_attempts < self.startup_context_samples:
            msg = "startup context attempts must cover the requested samples"
            raise ValueError(msg)
        if not 0.0 <= self.open_eye_threshold <= 1.0:
            msg = "open-eye threshold must be between zero and one"
            raise ValueError(msg)


class ConsoleStatus:
    """Report state transitions to standard error."""

    def report(self, status: RuntimeStatus, detail: str = "") -> None:
        """Write one line immediately without retaining it."""
        suffix = f": {detail}" if detail else ""
        sys.stderr.write(f"gazeebo: {status.value}{suffix}\n")
        sys.stderr.flush()


async def run_owned_session(  # noqa: C901, PLR0912, PLR0913, PLR0915
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    status: StatusSink,
    stop: asyncio.Event,
    *,
    hud: DebugHud | None = None,
    game: CalibrationGameSurface | None = None,
    tracking: TrackingConfig | None = None,
    game_config: GameConfig | None = None,
    training_store: TrainingStore | None = None,
    training_state: TrainingState | None = None,
    train_requested: bool = False,
    training_requested_event: asyncio.Event | None = None,
    game_factory: Callable[[Sequence[DisplayRegion]], CalibrationGameSurface] | None = None,
    training_control: TrainingControl | None = None,
    feature_schema: str = FEATURE_SCHEMA,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    """Select or train a model, navigate, and release all owned resources."""
    tracking_config = tracking or TrackingConfig()
    state = training_state or TrainingState()
    try:
        topology = DisplayTopology(pointer.regions)
        stored_router = _stored_router(
            state,
            topology,
            camera.camera_id,
            feature_schema,
            status,
        )
        incumbent: GazePredictor | None = stored_router
        model: GazePredictor
        initial_targets: tuple[CollectedTarget, ...] = ()

        if stored_router is None:
            if game is None:
                status.report(
                    RuntimeStatus.TRAINING_RECOMMENDED,
                    "no compatible stored model; initial training is required",
                )
            status.report(
                RuntimeStatus.INITIAL_TRAINING,
                f"{len(topology.regions)} authorized displays",
            )
            calibration = await _calibrate(
                camera,
                vision,
                pointer,
                topology,
                game,
                hud,
                status,
                stop,
                tracking_config,
                clock,
                sleep,
            )
            if stop.is_set() or calibration is None:
                return 0
            model = calibration.model
            base_samples = list(calibration.samples)
            initial_targets = calibration.targets
        else:
            model = stored_router
            base_samples = calibration_samples_for(
                state,
                topology,
                camera_id=camera.camera_id,
                feature_schema=feature_schema,
            )
            await _prime_router(
                camera,
                vision,
                model,
                status,
                stop,
                tracking_config,
                sleep,
            )

        should_train = game is not None and (train_requested or incumbent is None)
        if should_train and game is not None:
            game_result = await run_adaptive_game(
                camera,
                vision,
                pointer,
                topology,
                game,
                status,
                stop,
                base_samples,
                model,
                game_config or GameConfig(),
                incumbent_model=incumbent,
                force_adaptation=train_requested and incumbent is not None,
                target_offset=len(initial_targets),
                closed_threshold=tracking_config.open_eye_threshold,
                pointer_interval=tracking_config.pointer_update_interval_seconds,
                frame_interval=tracking_config.frame_interval_seconds,
                hud=hud,
                clock=clock,
                sleep=sleep,
            )
            await game.close()
            if stop.is_set() or game_result is None:
                return 0
            model = game_result.model
            pending = (*initial_targets, *game_result.accepted_targets)
            if training_store is not None and pending and game_result.persistent_accepted:
                persisted = _persist_targets(
                    training_store,
                    state,
                    topology,
                    camera.camera_id,
                    feature_schema,
                    pending,
                    game_result.after.median_error,
                    game_result.after.edge_error,
                    holdout=game_result.holdout_targets,
                    incumbent=incumbent,
                    validated_model=(
                        game_result.model
                        if isinstance(game_result.model, CalibrationModel)
                        else None
                    ),
                )
                if persisted is not None:
                    model, state = persisted
                else:
                    status.report(
                        RuntimeStatus.TRAINING_RECOMMENDED,
                        "persistent context model failed unseen acceptance",
                    )

        _set_hud_model_context(hud, model)
        while True:
            status.report(RuntimeStatus.ACTIVE)
            result = await _track(
                camera,
                vision,
                pointer,
                topology,
                model,
                hud,
                status,
                stop,
                tracking_config,
                training_requested_event,
                clock,
                sleep,
            )
            if result != TRAINING_REQUESTED_RESULT:
                return result
            if training_requested_event is not None:
                training_requested_event.clear()
            if game_factory is None:
                msg = "active training request has no target-surface factory"
                raise TrackingError(msg)
            active_game = game_factory(pointer.regions)
            try:
                base_samples = calibration_samples_for(
                    state,
                    topology,
                    camera_id=camera.camera_id,
                    feature_schema=feature_schema,
                )
                game_result = await run_adaptive_game(
                    camera,
                    vision,
                    pointer,
                    topology,
                    active_game,
                    status,
                    stop,
                    base_samples,
                    model,
                    game_config or GameConfig(),
                    incumbent_model=model,
                    force_adaptation=True,
                    closed_threshold=tracking_config.open_eye_threshold,
                    pointer_interval=tracking_config.pointer_update_interval_seconds,
                    frame_interval=tracking_config.frame_interval_seconds,
                    hud=hud,
                    clock=clock,
                    sleep=sleep,
                )
                if game_result is None:
                    return 0
                if (
                    training_store is not None
                    and game_result.accepted_targets
                    and game_result.persistent_accepted
                ):
                    persisted = _persist_targets(
                        training_store,
                        state,
                        topology,
                        camera.camera_id,
                        feature_schema,
                        game_result.accepted_targets,
                        game_result.after.median_error,
                        game_result.after.edge_error,
                        holdout=game_result.holdout_targets,
                        incumbent=model,
                        validated_model=(
                            game_result.model
                            if isinstance(game_result.model, CalibrationModel)
                            else None
                        ),
                    )
                    if persisted is not None:
                        model, state = persisted
                    else:
                        model = game_result.model
                        status.report(
                            RuntimeStatus.TRAINING_RECOMMENDED,
                            "persistent context model failed unseen acceptance",
                        )
                else:
                    model = game_result.model
                _set_hud_model_context(hud, model)
            finally:
                await active_game.close()
    finally:
        vision.close()
        camera.close()
        if game is not None:
            await game.close()
        if hud is not None:
            await hud.close()
        await pointer.close()
        if training_control is not None:
            await training_control.close()
        status.report(RuntimeStatus.STOPPED)


def _stored_router(
    state: TrainingState,
    topology: DisplayTopology,
    camera_id: str,
    feature_schema: str,
    status: StatusSink,
) -> ModelRouter | None:
    if not state.targets:
        return None
    status.report(RuntimeStatus.SELECTING_MODEL)
    try:
        router = build_router(
            state,
            topology,
            camera_id=camera_id,
            feature_schema=feature_schema,
        )
    except ValueError:
        return None
    quality = router.decide(state.targets[-1].context).topology_quality
    if quality is TopologyQuality.WEAK:
        status.report(
            RuntimeStatus.TOPOLOGY_UNVALIDATED,
            "using best-effort remapping and authorized-union clipping",
        )
    return router


async def _prime_router(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    router: ModelRouter,
    status: StatusSink,
    stop: asyncio.Event,
    tracking: TrackingConfig,
    sleep: Callable[[float], Awaitable[None]],
) -> None:
    """Choose an implicit context from a short passive startup window."""
    contexts: list[ContextVector] = []
    attempts = 0
    while (
        len(contexts) < tracking.startup_context_samples
        and attempts < tracking.startup_context_attempts
        and not stop.is_set()
    ):
        observation = vision.observe(camera.read(), time.monotonic())
        attempts += 1
        if observation is not None:
            contexts.append(observation.context)
        await sleep(tracking.frame_interval_seconds)
    if stop.is_set():
        return
    if len(contexts) < tracking.startup_context_samples:
        msg = "could not infer a stable startup context"
        raise TrackingError(msg)
    decision = router.decide(aggregate_features(contexts))
    status.report(
        RuntimeStatus.SELECTING_MODEL,
        f"{decision.label}; confidence {decision.confidence_label}",
    )
    if decision.out_of_distribution:
        status.report(
            RuntimeStatus.TRAINING_RECOMMENDED,
            "startup posture or illumination is outside learned contexts",
        )


async def _calibrate(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    game: CalibrationGameSurface | None,
    hud: DebugHud | None,
    status: StatusSink,
    stop: asyncio.Event,
    tracking: TrackingConfig,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> _BaseCalibration | None:
    samples: list[CalibrationSample] = []
    collected: list[CollectedTarget] = []
    targets = calibration_targets(topology)
    for target_index, target in enumerate(targets, start=1):
        if stop.is_set():
            break
        if game is None:
            pointer.move(target.region_id, target.x, target.y)
            await _update_hud(hud, topology, target)
        else:
            game.show_target(
                target.region_id,
                target.x,
                target.y,
                72.0,
                f"initial training {target_index}/{len(targets)}",
            )
        status.report(
            RuntimeStatus.INITIAL_TRAINING,
            f"target {target_index}/{len(targets)}",
        )
        await sleep(tracking.calibration_settle_seconds)
        target_features: list[FeatureVector] = []
        target_contexts: list[ContextVector] = []
        attempts = 0
        global_target = topology.to_global(target)
        while (
            len(target_features) < tracking.calibration_samples_per_target
            and attempts < tracking.calibration_attempts_per_target
            and not stop.is_set()
        ):
            observation = vision.observe(camera.read(), clock())
            attempts += 1
            if (
                observation is not None
                and observation.left_open > tracking.open_eye_threshold
                and observation.right_open > tracking.open_eye_threshold
            ):
                target_features.append(observation.features)
                target_contexts.append(observation.context)
            await sleep(tracking.frame_interval_seconds)
        if not stop.is_set():
            if len(target_features) < tracking.calibration_samples_per_target:
                msg = f"could not observe open eyes for initial target {target_index}"
                raise TrackingError(msg)
            features = aggregate_features(target_features)
            context = aggregate_features(target_contexts)
            samples.append(CalibrationSample(features, global_target))
            collected.append(
                CollectedTarget(
                    features,
                    context,
                    target,
                    _calibration_zone(topology, target),
                )
            )
    if stop.is_set():
        return None
    return _BaseCalibration(
        CalibrationModel.fit(samples),
        tuple(samples),
        tuple(collected),
    )


def _persist_targets(  # noqa: PLR0913
    store: TrainingStore,
    existing: TrainingState,
    topology: DisplayTopology,
    camera_id: str,
    feature_schema: str,
    targets: Sequence[CollectedTarget],
    median_error: float,
    edge_error: float,
    *,
    holdout: Sequence[CollectedTarget] = (),
    incumbent: GazePredictor | None = None,
    validated_model: CalibrationModel | None = None,
) -> tuple[ModelRouter, TrainingState] | None:
    """Build and atomically commit one accepted persistent candidate."""
    candidate = copy.deepcopy(existing)
    assigned_clusters: list[str] = []
    for collected in targets:
        persistent = make_stored_target(
            candidate.next_sequence,
            camera_id,
            feature_schema,
            collected.features,
            collected.context,
            topology,
            collected.target,
            collected.zone,
        )
        assigned_clusters.append(add_target(candidate, persistent))
    router = build_router(
        candidate,
        topology,
        camera_id=camera_id,
        feature_schema=feature_schema,
    )
    dominant_cluster = (
        Counter(assigned_clusters).most_common(1)[0][0] if assigned_clusters else None
    )
    if validated_model is not None and dominant_cluster is not None:
        router = router.with_validated_model(
            validated_model,
            dominant_cluster,
            replace_global=incumbent is None,
        )
    if holdout:
        candidate_metrics = _score_collected(router, holdout, topology)
        incumbent_metrics = (
            None if incumbent is None else _score_collected(incumbent, holdout, topology)
        )
        if not all(
            math.isfinite(value)
            for value in (candidate_metrics.median_error, candidate_metrics.edge_error)
        ):
            return None
        if incumbent is not None and not candidate_is_acceptable(
            incumbent_metrics,
            candidate_metrics,
        ):
            return None
        median_error = candidate_metrics.median_error
        edge_error = candidate_metrics.edge_error
    if dominant_cluster is not None:
        candidate.clusters = [
            replace(
                cluster,
                median_error=median_error,
                edge_error=edge_error,
            )
            if cluster.cluster_id == dominant_cluster
            else cluster
            for cluster in candidate.clusters
        ]
    prefix = f"{camera_id}:{topology.topology_id}:"
    candidate.models = {
        key: value for key, value in candidate.models.items() if not key.startswith(prefix)
    }
    candidate.models.update({f"{prefix}{name}": value for name, value in router.records().items()})
    decision = router.decide(targets[-1].context)
    candidate.validations.append(
        ValidationSummary(
            candidate.next_sequence,
            camera_id,
            topology.topology_id,
            decision.label,
            median_error,
            edge_error,
        )
    )
    candidate.validations = candidate.validations[-64:]
    store.save(candidate)
    return router, candidate


def _score_collected(
    model: GazePredictor,
    targets: Sequence[CollectedTarget],
    topology: DisplayTopology,
) -> ValidationMetrics:
    errors: list[float] = []
    edge_errors: list[float] = []
    for target in targets:
        predicted = topology.to_global(
            topology.locate(model.predict(target.features, target.context))
        )
        expected = topology.to_global(target.target)
        error = math.hypot(predicted.x - expected.x, predicted.y - expected.y)
        errors.append(error)
        if target.zone != "center":
            edge_errors.append(error)
    return ValidationMetrics(
        statistics.median(errors),
        statistics.median(edge_errors or errors),
    )


async def _track(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    calibration: GazePredictor,
    hud: DebugHud | None,
    status: StatusSink,
    stop: asyncio.Event,
    tracking: TrackingConfig,
    training_requested: asyncio.Event | None,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> int:
    smoother = PointerSmoother(
        alpha=tracking.smoothing_alpha,
        dead_zone=tracking.smoothing_dead_zone,
        maximum_step=tracking.smoothing_maximum_step,
    )
    missed_frames = 0
    last_pointer_update: float | None = None
    while not stop.is_set():
        if training_requested is not None and training_requested.is_set():
            return TRAINING_REQUESTED_RESULT
        if bool(getattr(pointer, "closed", False)):
            status.report(RuntimeStatus.RECALIBRATION_REQUIRED, "desktop authorization closed")
            return 3
        observation = vision.observe(camera.read(), clock())
        if observation is None:
            missed_frames += 1
            if missed_frames >= tracking.maximum_missed_frames:
                msg = "face or eyes remained unavailable"
                raise TrackingError(msg)
            await sleep(tracking.frame_interval_seconds)
            continue
        missed_frames = 0
        pointer_due = (
            last_pointer_update is None
            or observation.timestamp - last_pointer_update
            >= tracking.pointer_update_interval_seconds
        )
        eyes_open = (
            observation.left_open > tracking.open_eye_threshold
            and observation.right_open > tracking.open_eye_threshold
        )
        if pointer_due and eyes_open:
            prediction = calibration.predict(observation.features, observation.context)
            target = topology.locate(smoother.update(prediction))
            pointer.move(target.region_id, target.x, target.y)
            await _update_hud(hud, topology, target)
            last_pointer_update = observation.timestamp
        await sleep(tracking.frame_interval_seconds)
    return 0


def _calibration_zone(topology: DisplayTopology, target: PointerTarget) -> str:
    region = topology.region(target.region_id)
    horizontal = target.x / region.width
    vertical = target.y / region.height
    far_edge = 1.0 - CALIBRATION_EDGE_BOUNDARY
    x_edge = horizontal < CALIBRATION_EDGE_BOUNDARY or horizontal > far_edge
    y_edge = vertical < CALIBRATION_EDGE_BOUNDARY or vertical > far_edge
    if x_edge and y_edge:
        return "corner"
    if x_edge or y_edge:
        return "edge"
    return "center"


def _set_hud_model_context(
    hud: DebugHud | None,
    model: GazePredictor,
) -> None:
    if hud is None:
        return
    if isinstance(model, ModelRouter) and model.last_decision is not None:
        decision = model.last_decision
        hud.set_model_context(
            decision.label,
            decision.topology_quality.name.lower(),
            decision.confidence_label,
        )
    else:
        hud.set_model_context(model.kind, "current-session", "session-only")


async def _update_hud(
    hud: DebugHud | None,
    topology: DisplayTopology,
    target: PointerTarget,
) -> None:
    """Publish one global coordinate when diagnostics are enabled."""
    if hud is None:
        return
    point = topology.to_global(target)
    await hud.update(target.region_id, point.x, point.y)


def install_signal_handlers(stop: asyncio.Event) -> None:
    """Map interactive and release-triggered termination onto one stop event."""
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
