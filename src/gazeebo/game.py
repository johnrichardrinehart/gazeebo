"""Iterative adaptive calibration game and unseen-batch metrics."""

from __future__ import annotations

import ctypes
import math
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from gazeebo.calibration import (
    CalibrationModel,
    CalibrationSample,
    aggregate_features,
)
from gazeebo.contexts import ValidationMetrics, candidate_is_acceptable
from gazeebo.contracts import RuntimeStatus
from gazeebo.geometry import DisplayTopology, Point, PointerTarget
from gazeebo.native import NativeRendererError, load_native_renderer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from gazeebo.contracts import (
        CalibrationGameSurface,
        CameraCapture,
        DebugHud,
        DisplayRegion,
        EyeObservation,
        FeatureVector,
        PointerController,
        StatusSink,
        VisionEstimator,
    )

GAME_ERROR_SIZE = 256
MINIMUM_GAME_TARGETS = 5
MINIMUM_STABLE_SAMPLES = 4
MAXIMUM_ADAPTIVE_SAMPLES = 25
EDGE_BOUNDARY = 0.18

_TARGET_POSITIONS = (
    (0.16, 0.16),
    (0.84, 0.16),
    (0.50, 0.50),
    (0.16, 0.84),
    (0.84, 0.84),
    (0.50, 0.10),
    (0.90, 0.50),
    (0.50, 0.90),
    (0.10, 0.50),
    (0.32, 0.32),
    (0.68, 0.32),
    (0.32, 0.68),
    (0.68, 0.68),
    (0.25, 0.12),
    (0.75, 0.88),
)


class GameError(RuntimeError):
    """Adaptive calibration cannot continue safely."""


class GazePredictor(Protocol):
    """Predict global coordinates with optional context routing."""

    @property
    def kind(self) -> str:
        """Return a concise estimator label."""

    def predict(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> Point:
        """Return one global logical prediction."""


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Finite adaptive-game timing, target, and stability controls."""

    batch_size: int = 5
    maximum_targets: int = 55
    precision_threshold: float = 100.0
    settle_seconds: float = 0.75
    dwell_seconds: float = 1.25
    target_timeout_seconds: float = 6.0
    minimum_diameter: float = 48.0
    maximum_diameter: float = 144.0
    stability_threshold: float = 0.08
    minimum_confidence: float = 0.35

    def __post_init__(self) -> None:  # noqa: C901
        """Reject invalid or non-terminating game settings."""
        if self.batch_size < MINIMUM_GAME_TARGETS:
            msg = f"game batch size must be at least {MINIMUM_GAME_TARGETS}"
            raise ValueError(msg)
        if self.maximum_targets < self.batch_size:
            msg = "game maximum must cover one complete batch"
            raise ValueError(msg)
        if self.maximum_targets % self.batch_size != 0:
            msg = "game maximum must be a multiple of batch size"
            raise ValueError(msg)
        finite_values = (
            self.precision_threshold,
            self.settle_seconds,
            self.dwell_seconds,
            self.target_timeout_seconds,
            self.minimum_diameter,
            self.maximum_diameter,
            self.stability_threshold,
            self.minimum_confidence,
        )
        if not all(math.isfinite(value) for value in finite_values):
            msg = "game settings must be finite"
            raise ValueError(msg)
        if self.precision_threshold <= 0.0:
            msg = "game precision threshold must be positive"
            raise ValueError(msg)
        if self.settle_seconds < 0.0 or self.dwell_seconds <= 0.0:
            msg = "game settle must be non-negative and dwell must be positive"
            raise ValueError(msg)
        if self.target_timeout_seconds < self.dwell_seconds:
            msg = "game target timeout must cover the dwell interval"
            raise ValueError(msg)
        if not 0.0 < self.minimum_diameter <= self.maximum_diameter:
            msg = "game circle diameters must be positive and ordered"
            raise ValueError(msg)
        if self.stability_threshold <= 0.0:
            msg = "game stability threshold must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.minimum_confidence <= 1.0:
            msg = "game confidence must be between zero and one"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class GameTarget:
    """One region-local visual target."""

    region_id: str
    x: float
    y: float
    diameter: float
    edge_or_corner: bool


@dataclass(frozen=True, slots=True)
class TargetMeasurement:
    """Accuracy measured while looking at one holdout target."""

    radial_error: float
    edge_or_corner: bool
    response_seconds: float | None


@dataclass(frozen=True, slots=True)
class GameMetrics:
    """Aggregate non-persistent validation measurements."""

    target_count: int
    hit_count: int
    median_error: float
    edge_error: float
    median_response: float | None

    def summary(self, label: str) -> str:
        """Format one concise status line without persisting measurements."""
        response = f"{self.median_response:.2f}s" if self.median_response is not None else "n/a"
        return (
            f"{label}: hits {self.hit_count}/{self.target_count}, "
            f"median error {self.median_error:.0f}px, "
            f"edge error {self.edge_error:.0f}px, response {response}"
        )


@dataclass(frozen=True, slots=True)
class CollectedTarget:
    """One reported target aggregate eligible for later training."""

    features: FeatureVector
    context: tuple[float, ...]
    target: PointerTarget
    zone: str


@dataclass(frozen=True, slots=True)
class GameResult:
    """Improved model, accepted training aggregates, and holdout evidence."""

    model: GazePredictor
    rounds: tuple[GameMetrics, ...]
    precision_met: bool
    accepted_targets: tuple[CollectedTarget, ...] = ()
    holdout_targets: tuple[CollectedTarget, ...] = ()
    persistent_accepted: bool = True

    @property
    def before(self) -> GameMetrics:
        """Return the first unseen-batch metric for comparisons."""
        return self.rounds[0]

    @property
    def after(self) -> GameMetrics:
        """Return the final unseen-batch metric for comparisons."""
        return self.rounds[-1]


@dataclass(frozen=True, slots=True)
class _TargetResult:
    features: tuple[FeatureVector, ...]
    contexts: tuple[tuple[float, ...], ...]
    measurement: TargetMeasurement
    incumbent_measurement: TargetMeasurement


@dataclass(frozen=True, slots=True)
class _ValidationResult:
    metrics: GameMetrics
    incumbent_metrics: GameMetrics
    samples: tuple[CalibrationSample, ...]
    targets: tuple[CollectedTarget, ...]


@dataclass(slots=True)
class _PointerCadence:
    interval: float
    last_update: float | None = None
    visible_point: Point | None = None

    def due(self, timestamp: float) -> bool:
        if self.last_update is None or timestamp - self.last_update >= self.interval:
            self.last_update = timestamp
            return True
        return False


class _NativeGameSurface:
    """Bind the adaptive game to the packaged native Wayland renderer."""

    def __init__(self, region: DisplayRegion) -> None:
        try:
            library = load_native_renderer()
        except NativeRendererError as load_error:
            raise GameError(str(load_error)) from load_error
        library.gazeebo_game_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_game_create.restype = ctypes.c_void_p
        library.gazeebo_game_show_target.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_game_show_target.restype = ctypes.c_int
        library.gazeebo_game_hide.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        library.gazeebo_game_hide.restype = ctypes.c_int
        library.gazeebo_game_destroy.argtypes = [ctypes.c_void_p]
        library.gazeebo_game_destroy.restype = None
        error_buffer = ctypes.create_string_buffer(GAME_ERROR_SIZE)
        handle = library.gazeebo_game_create(
            region.x,
            region.y,
            region.width,
            region.height,
            error_buffer,
            GAME_ERROR_SIZE,
        )
        if not handle:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration game failed to start: {detail}"
            raise GameError(msg)
        self._library = library
        self._handle = ctypes.c_void_p(handle)
        self._closed = False

    def show_target(self, x: float, y: float, diameter: float, label: str) -> None:
        if self._closed:
            return
        error_buffer = ctypes.create_string_buffer(GAME_ERROR_SIZE)
        result = self._library.gazeebo_game_show_target(
            self._handle,
            x,
            y,
            diameter,
            label.encode(),
            error_buffer,
            GAME_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration game update failed: {detail}"
            raise GameError(msg)

    def hide(self) -> None:
        if self._closed:
            return
        error_buffer = ctypes.create_string_buffer(GAME_ERROR_SIZE)
        result = self._library.gazeebo_game_hide(
            self._handle,
            error_buffer,
            GAME_ERROR_SIZE,
        )
        if result != 0:
            detail = error_buffer.value.decode(errors="replace") or "unknown Wayland error"
            msg = f"calibration game hide failed: {detail}"
            raise GameError(msg)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._library.gazeebo_game_destroy(self._handle)


class LayerShellCalibrationGame:
    """Own click-through game surfaces on all authorized displays."""

    def __init__(self, surfaces: dict[str, _NativeGameSurface]) -> None:
        """Own the supplied native surfaces until explicit cleanup."""
        self._surfaces = surfaces
        self._closed = False

    @classmethod
    def create(cls, regions: Sequence[DisplayRegion]) -> LayerShellCalibrationGame:
        """Create one native game surface per authorized portal region."""
        surfaces: dict[str, _NativeGameSurface] = {}
        try:
            for region in regions:
                surfaces[region.region_id] = _NativeGameSurface(region)
        except Exception:
            for surface in surfaces.values():
                surface.close()
            raise
        return cls(surfaces)

    def show_target(
        self,
        region_id: str,
        x: float,
        y: float,
        diameter: float,
        label: str,
    ) -> None:
        """Show one target and clear every other authorized display."""
        if self._closed:
            return
        target_surface = self._surfaces.get(region_id)
        if target_surface is None:
            msg = f"unknown calibration game region: {region_id}"
            raise GameError(msg)
        for identifier, surface in self._surfaces.items():
            if identifier == region_id:
                surface.show_target(x, y, diameter, label)
            else:
                surface.hide()

    async def close(self) -> None:
        """Destroy every native surface idempotently."""
        if self._closed:
            return
        self._closed = True
        for surface in self._surfaces.values():
            surface.close()


def game_targets(
    regions: Sequence[DisplayRegion],
    count: int,
    minimum_diameter: float,
    maximum_diameter: float,
    *,
    start_index: int = 0,
) -> tuple[GameTarget, ...]:
    """Generate deterministic unseen targets across authorized displays."""
    if not regions:
        msg = "game requires at least one authorized display"
        raise ValueError(msg)
    if count <= 0 or start_index < 0:
        msg = "game target count and start index must be non-negative"
        raise ValueError(msg)
    if any(maximum_diameter >= min(region.width, region.height) for region in regions):
        msg = "game circles must fit within every authorized display"
        raise GameError(msg)
    targets: list[GameTarget] = []
    for index in range(start_index, start_index + count):
        region = regions[index % len(regions)]
        position_index = index // len(regions)
        if position_index < len(_TARGET_POSITIONS):
            normalized_x, normalized_y = _TARGET_POSITIONS[position_index]
        else:
            offset = position_index + 11
            normalized_x = 0.08 + 0.84 * _halton(offset, 2)
            normalized_y = 0.08 + 0.84 * _halton(offset, 3)
        diameter_fraction = (index % 5) / 4.0
        diameter = minimum_diameter + (maximum_diameter - minimum_diameter) * diameter_fraction
        radius = diameter / 2.0
        x = radius + normalized_x * (region.width - diameter)
        y = radius + normalized_y * (region.height - diameter)
        edge = (
            normalized_x <= EDGE_BOUNDARY
            or normalized_x >= 1.0 - EDGE_BOUNDARY
            or normalized_y <= EDGE_BOUNDARY
            or normalized_y >= 1.0 - EDGE_BOUNDARY
        )
        targets.append(GameTarget(region.region_id, x, y, diameter, edge))
    return tuple(targets)


def _halton(index: int, base: int) -> float:
    result = 0.0
    fraction = 1.0
    while index > 0:
        fraction /= base
        result += fraction * (index % base)
        index //= base
    return result


def game_metrics(measurements: Sequence[TargetMeasurement]) -> GameMetrics:
    """Aggregate holdout target measurements deterministically."""
    if not measurements:
        msg = "validation requires at least one target measurement"
        raise ValueError(msg)
    errors = [item.radial_error for item in measurements]
    edge_errors = [item.radial_error for item in measurements if item.edge_or_corner]
    responses = [
        item.response_seconds for item in measurements if item.response_seconds is not None
    ]
    return GameMetrics(
        target_count=len(measurements),
        hit_count=len(responses),
        median_error=statistics.median(errors),
        edge_error=statistics.median(edge_errors or errors),
        median_response=statistics.median(responses) if responses else None,
    )


async def run_adaptive_game(  # noqa: PLR0913, PLR0915
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    surface: CalibrationGameSurface,
    status: StatusSink,
    stop: object,
    base_samples: Sequence[CalibrationSample],
    initial_model: GazePredictor,
    config: GameConfig,
    *,
    incumbent_model: GazePredictor | None = None,
    force_adaptation: bool = False,
    target_offset: int = 0,
    closed_threshold: float,
    pointer_interval: float,
    frame_interval: float,
    hud: DebugHud | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]],
) -> GameResult | None:
    """Evaluate unseen batches, adapt after failures, and stop at precision."""
    cadence = _PointerCadence(pointer_interval)
    samples = list(base_samples)
    model = initial_model
    incumbent = incumbent_model or initial_model
    accepted_targets: list[CollectedTarget] = []
    rounds: list[GameMetrics] = []
    best_model: GazePredictor | None = None
    best_result: _ValidationResult | None = None
    best_batch = 0
    best_score = (math.inf, math.inf)
    if target_offset < 0 or target_offset % config.batch_size != 0:
        msg = "game target offset must be a non-negative batch multiple"
        raise ValueError(msg)
    if target_offset >= config.maximum_targets:
        msg = "game target offset must leave room for one unseen batch"
        raise ValueError(msg)
    presented = 0
    while target_offset + presented < config.maximum_targets:
        batch_number = len(rounds) + 1
        targets = game_targets(
            topology.regions,
            config.batch_size,
            config.minimum_diameter,
            config.maximum_diameter,
            start_index=target_offset + presented,
        )
        status.report(
            RuntimeStatus.GAME_VALIDATING,
            f"unseen batch {batch_number}",
        )
        result = await _validate(
            camera,
            vision,
            pointer,
            topology,
            surface,
            targets,
            model,
            incumbent,
            config,
            cadence,
            stop,
            closed_threshold,
            frame_interval,
            hud,
            clock,
            sleep,
            f"batch {batch_number}",
        )
        if result is None:
            return None
        presented += len(targets)
        rounds.append(result.metrics)
        status.report(
            RuntimeStatus.GAME_VALIDATING,
            (
                f"{result.metrics.summary(f'batch {batch_number}')}, "
                f"estimator {model.kind}, "
                f"total {target_offset + presented}/{config.maximum_targets}"
            ),
        )
        precision_met = (
            result.metrics.median_error <= config.precision_threshold
            and result.metrics.edge_error <= config.precision_threshold
        )
        accepted = candidate_is_acceptable(
            ValidationMetrics(
                result.incumbent_metrics.median_error,
                result.incumbent_metrics.edge_error,
            ),
            ValidationMetrics(
                result.metrics.median_error,
                result.metrics.edge_error,
            ),
        )
        must_adapt = force_adaptation and not accepted_targets
        score = (
            max(result.metrics.median_error, result.metrics.edge_error),
            result.metrics.median_error + result.metrics.edge_error,
        )
        if accepted and not must_adapt and score < best_score:
            best_model = model
            best_result = result
            best_batch = batch_number
            best_score = score
        if precision_met and accepted and not must_adapt:
            return GameResult(
                model,
                tuple(rounds),
                precision_met=True,
                accepted_targets=tuple(accepted_targets),
                holdout_targets=result.targets,
                persistent_accepted=True,
            )
        if precision_met:
            status.report(
                RuntimeStatus.GAME_TRAINING,
                "candidate rejected because unseen incumbent comparison regressed",
            )
        if target_offset + presented >= config.maximum_targets:
            message = (
                f"precision target {config.precision_threshold:.0f}px not met "
                f"after {target_offset + presented} circles"
            )
            status.report(RuntimeStatus.TRAINING_RECOMMENDED, message)
            if best_model is None or best_result is None:
                return GameResult(
                    incumbent,
                    tuple(rounds),
                    precision_met=False,
                    persistent_accepted=False,
                )
            status.report(
                RuntimeStatus.GAME_TRAINING,
                (
                    f"retaining batch {best_batch} model with "
                    f"{best_result.metrics.median_error:.0f}px median and "
                    f"{best_result.metrics.edge_error:.0f}px edge error"
                ),
            )
            best_holdout = set(best_result.targets)
            return GameResult(
                best_model,
                tuple(rounds),
                precision_met=False,
                accepted_targets=tuple(
                    target for target in accepted_targets if target not in best_holdout
                ),
                holdout_targets=best_result.targets,
                persistent_accepted=True,
            )
        status.report(
            RuntimeStatus.GAME_TRAINING,
            f"adapting from reported batch {batch_number}",
        )
        samples.extend(result.samples)
        accepted_targets.extend(result.targets)
        model = CalibrationModel.fit(samples[-MAXIMUM_ADAPTIVE_SAMPLES:])
    msg = "adaptive calibration ended without a terminal batch"
    raise GameError(msg)


async def _validate(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    surface: CalibrationGameSurface,
    targets: Sequence[GameTarget],
    model: GazePredictor,
    incumbent: GazePredictor,
    config: GameConfig,
    cadence: _PointerCadence,
    stop: object,
    closed_threshold: float,
    frame_interval: float,
    hud: DebugHud | None,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    phase: str,
) -> _ValidationResult | None:
    measurements: list[TargetMeasurement] = []
    incumbent_measurements: list[TargetMeasurement] = []
    samples: list[CalibrationSample] = []
    collected: list[CollectedTarget] = []
    for index, target in enumerate(targets, start=1):
        if _stop_is_set(stop):
            return None
        surface.show_target(
            target.region_id,
            target.x,
            target.y,
            target.diameter,
            f"{phase} {index}/{len(targets)}",
        )
        result = await _collect_target(
            camera,
            vision,
            pointer,
            topology,
            target,
            model,
            incumbent,
            config,
            cadence,
            stop,
            closed_threshold,
            frame_interval,
            hud,
            clock,
            sleep,
        )
        if result is None:
            return None
        measurements.append(result.measurement)
        incumbent_measurements.append(result.incumbent_measurement)
        features = aggregate_features(result.features)
        context = aggregate_features(result.contexts)
        pointer_target = PointerTarget(target.region_id, target.x, target.y)
        samples.append(
            CalibrationSample(
                features,
                topology.to_global(pointer_target),
            )
        )
        collected.append(
            CollectedTarget(
                features,
                context,
                pointer_target,
                _target_zone(target, topology),
            )
        )
    return _ValidationResult(
        game_metrics(measurements),
        game_metrics(incumbent_measurements),
        tuple(samples),
        tuple(collected),
    )


async def _collect_target(  # noqa: PLR0913
    camera: CameraCapture,
    vision: VisionEstimator,
    pointer: PointerController,
    topology: DisplayTopology,
    target: GameTarget,
    model: GazePredictor,
    incumbent: GazePredictor,
    config: GameConfig,
    cadence: _PointerCadence,
    stop: object,
    closed_threshold: float,
    frame_interval: float,
    hud: DebugHud | None,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> _TargetResult | None:
    await sleep(config.settle_seconds)
    started = clock()
    first_hit: float | None = None
    window: list[tuple[float, EyeObservation, Point, float, float]] = []
    target_global = topology.to_global(PointerTarget(target.region_id, target.x, target.y))
    while clock() - started <= config.target_timeout_seconds:
        if _stop_is_set(stop):
            return None
        timestamp = clock()
        observation = vision.observe(camera.read(), timestamp)
        if (
            observation is not None
            and observation.confidence >= config.minimum_confidence
            and observation.left_open > closed_threshold
            and observation.right_open > closed_threshold
        ):
            raw = model.predict(observation.features, observation.context)
            incumbent_raw = incumbent.predict(observation.features, observation.context)
            pointer_target = topology.locate(raw)
            clipped = topology.to_global(pointer_target)
            incumbent_clipped = topology.to_global(topology.locate(incumbent_raw))
            if cadence.due(observation.timestamp):
                pointer.move(pointer_target.region_id, pointer_target.x, pointer_target.y)
                cadence.visible_point = clipped
                if hud is not None:
                    await hud.update(pointer_target.region_id, clipped.x, clipped.y)
            visible = cadence.visible_point or clipped
            error = math.hypot(clipped.x - target_global.x, clipped.y - target_global.y)
            visible_error = math.hypot(
                visible.x - target_global.x,
                visible.y - target_global.y,
            )
            incumbent_error = math.hypot(
                incumbent_clipped.x - target_global.x,
                incumbent_clipped.y - target_global.y,
            )
            if first_hit is None and visible_error <= target.diameter / 2.0:
                first_hit = max(0.0, observation.timestamp - started)
            window.append((observation.timestamp, observation, raw, error, incumbent_error))
            cutoff = observation.timestamp - config.dwell_seconds
            window = [item for item in window if item[0] >= cutoff]
            if (
                observation.timestamp - started >= config.dwell_seconds
                and len(window) >= MINIMUM_STABLE_SAMPLES
                and _stable_observations(window, config.stability_threshold)
            ):
                return _TargetResult(
                    tuple(item[1].features for item in window),
                    tuple(item[1].context for item in window),
                    TargetMeasurement(
                        radial_error=statistics.median(item[3] for item in window),
                        edge_or_corner=target.edge_or_corner,
                        response_seconds=first_hit,
                    ),
                    TargetMeasurement(
                        radial_error=statistics.median(item[4] for item in window),
                        edge_or_corner=target.edge_or_corner,
                        response_seconds=None,
                    ),
                )
        await sleep(frame_interval)
    msg = "could not obtain stable open-eye samples for a game target"
    raise GameError(msg)


def _target_zone(target: GameTarget, topology: DisplayTopology) -> str:
    """Classify a persistent target for balanced center/edge/corner retention."""
    region = topology.region(target.region_id)
    horizontal_edge = (
        target.x / region.width <= EDGE_BOUNDARY or target.x / region.width >= 1.0 - EDGE_BOUNDARY
    )
    vertical_edge = (
        target.y / region.height <= EDGE_BOUNDARY or target.y / region.height >= 1.0 - EDGE_BOUNDARY
    )
    if horizontal_edge and vertical_edge:
        return "corner"
    if horizontal_edge or vertical_edge:
        return "edge"
    return "center"


def _stable_observations(
    window: Sequence[tuple[float, EyeObservation, Point, float, float]],
    maximum_deviation: float,
) -> bool:
    feature_count = len(window[0][1].features)
    deviations = [
        statistics.pstdev(item[1].features[index] for item in window)
        for index in range(feature_count)
    ]
    root_mean_square = math.sqrt(statistics.fmean(value**2 for value in deviations))
    return root_mean_square <= maximum_deviation


def _stop_is_set(stop: object) -> bool:
    method = getattr(stop, "is_set", None)
    return bool(method()) if callable(method) else False
