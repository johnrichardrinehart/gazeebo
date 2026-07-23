"""Bounded context clustering and global/local gaze-model routing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gazeebo.adaptation import TopologyQuality, map_stored_target
from gazeebo.calibration import CalibrationModel, CalibrationSample
from gazeebo.geometry import Point
from gazeebo.state import ContextCluster, StoredTarget, TrainingState

MINIMUM_MODEL_TARGETS = 3
MINIMUM_ROUTING_WEIGHT = 1e-6
ROUTING_LABEL_WEIGHT = 0.05

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from gazeebo.contracts import FeatureVector
    from gazeebo.geometry import DisplayTopology


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Finite clustering, retention, and routing policy."""

    maximum_clusters_per_partition: int = 8
    maximum_cluster_targets: int = 64
    maximum_global_targets: int = 256
    assignment_distance: float = 2.5
    merge_distance: float = 1.25
    variance_floor: float = 0.0025
    routing_smoothing: float = 0.25
    switching_margin: float = 0.15

    def __post_init__(self) -> None:
        """Reject unbounded or numerically invalid context policy."""
        counts = (
            self.maximum_clusters_per_partition,
            self.maximum_cluster_targets,
            self.maximum_global_targets,
        )
        if any(value <= 0 for value in counts):
            msg = "context limits must be positive"
            raise ValueError(msg)
        if (
            self.assignment_distance <= 0.0
            or self.merge_distance < 0.0
            or self.variance_floor <= 0.0
            or not 0.0 < self.routing_smoothing <= 1.0
            or self.switching_margin < 0.0
        ):
            msg = "context distances and routing settings are invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ValidationMetrics:
    """The two accuracy gates used for persistent candidate acceptance."""

    median_error: float
    edge_error: float

    def __post_init__(self) -> None:
        """Reject invalid validation errors."""
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.median_error, self.edge_error)
        ):
            msg = "validation metrics must be finite and non-negative"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ContextExpert:
    """One local estimator and its routing statistics."""

    cluster: ContextCluster
    model: CalibrationModel


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Smoothed weights and quality exposed to runtime status and the HUD."""

    weights: tuple[tuple[str, float], ...]
    topology_quality: TopologyQuality
    out_of_distribution: bool

    @property
    def label(self) -> str:
        """Return a stable human-readable blend label."""
        return "+".join(name for name, weight in self.weights if weight >= ROUTING_LABEL_WEIGHT)


class ModelRouter:
    """Blend a global estimator with compatible context experts."""

    def __init__(  # noqa: PLR0913
        self,
        global_model: CalibrationModel,
        experts: Sequence[ContextExpert],
        *,
        camera_id: str,
        feature_schema: str,
        topology_quality: TopologyQuality,
        config: ContextConfig | None = None,
    ) -> None:
        """Bind compatible estimators to one implicit camera/schema partition."""
        self._global = global_model
        self._experts = tuple(experts)
        self._camera_id = camera_id
        self._feature_schema = feature_schema
        self._quality = topology_quality
        self._config = config or ContextConfig()
        self._weights: dict[str, float] = {"global": 1.0}
        self._last_best = "global"
        self._initialized = False
        self._last_decision: RoutingDecision | None = None

    def decide(self, context: tuple[float, ...]) -> RoutingDecision:
        """Update context weights with a switching margin and EWMA smoothing."""
        compatible: list[tuple[ContextExpert, float, float]] = []
        for expert in self._experts:
            cluster = expert.cluster
            if (
                cluster.camera_id != self._camera_id
                or cluster.feature_schema != self._feature_schema
            ):
                continue
            distance = context_distance(context, cluster, self._config.variance_floor)
            quality = _validation_weight(cluster)
            compatible.append((expert, distance, math.exp(-0.5 * distance**2) * quality))
        compatible.sort(key=lambda item: (-item[2], item[0].cluster.cluster_id))
        selected = compatible[:2]
        nearest = min((item[1] for item in selected), default=math.inf)
        out_of_distribution = nearest > self._config.assignment_distance
        raw: dict[str, float] = {
            "global": 1.0
            if out_of_distribution
            else max(0.15, nearest / self._config.assignment_distance)
        }
        for expert, _distance, score in selected:
            raw[expert.cluster.cluster_id] = score

        best = max(raw, key=raw.__getitem__)
        previous_score = raw.get(self._last_best, 0.0)
        if best != self._last_best and raw[best] < previous_score * (
            1.0 + self._config.switching_margin
        ):
            raw[self._last_best] = max(raw[best], previous_score)
            best = self._last_best
        self._last_best = best

        names = set(self._weights) | set(raw)
        alpha = self._config.routing_smoothing
        if self._initialized:
            smoothed = {
                name: (1.0 - alpha) * self._weights.get(name, 0.0) + alpha * raw.get(name, 0.0)
                for name in names
            }
        else:
            smoothed = raw
            self._initialized = True
        total = sum(smoothed.values())
        self._weights = {
            name: value / total
            for name, value in smoothed.items()
            if value > MINIMUM_ROUTING_WEIGHT
        }
        ordered = tuple(sorted(self._weights.items(), key=lambda item: (-item[1], item[0])))
        decision = RoutingDecision(ordered, self._quality, out_of_distribution)
        self._last_decision = decision
        return decision

    @property
    def kind(self) -> str:
        """Describe the adaptive estimator family."""
        return "context-mixture"

    @property
    def last_decision(self) -> RoutingDecision | None:
        """Expose the most recent routing result for status and diagnostics."""
        return self._last_decision

    def predict(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> Point:
        """Blend current model predictions using smoothed context routing."""
        if context is None:
            msg = "context model requires a routing context"
            raise ValueError(msg)
        point, _decision = self.predict_with_decision(features, context)
        return point

    def predict_with_decision(
        self,
        features: FeatureVector,
        context: tuple[float, ...],
    ) -> tuple[Point, RoutingDecision]:
        """Return a blended prediction and its observable routing decision."""
        decision = self.decide(context)
        models = {expert.cluster.cluster_id: expert.model for expert in self._experts}
        models["global"] = self._global
        x = 0.0
        y = 0.0
        used = 0.0
        for name, weight in decision.weights:
            model = models.get(name)
            if model is None:
                continue
            point = model.predict(features)
            x += point.x * weight
            y += point.y * weight
            used += weight
        if used <= 0.0:
            msg = "model router has no usable estimator"
            raise RuntimeError(msg)
        self._last_decision = decision
        return Point(x / used, y / used), decision

    def with_validated_model(
        self,
        model: CalibrationModel,
        cluster_id: str,
        *,
        replace_global: bool,
    ) -> ModelRouter:
        """Use the exact model measured by the terminal unseen batch."""
        experts = tuple(
            ContextExpert(expert.cluster, model)
            if expert.cluster.cluster_id == cluster_id
            else expert
            for expert in self._experts
        )
        return ModelRouter(
            model if replace_global else self._global,
            experts,
            camera_id=self._camera_id,
            feature_schema=self._feature_schema,
            topology_quality=self._quality,
            config=self._config,
        )

    def records(self) -> dict[str, dict[str, object]]:
        """Return serializable global and expert coefficients."""
        records = {"global": self._global.to_record()}
        records.update(
            {expert.cluster.cluster_id: expert.model.to_record() for expert in self._experts}
        )
        return records


def context_distance(
    context: tuple[float, ...],
    cluster: ContextCluster,
    variance_floor: float,
) -> float:
    """Return diagonal-variance normalized context distance."""
    if len(context) != len(cluster.centroid):
        return math.inf
    difference = np.asarray(context) - np.asarray(cluster.centroid)
    scale = np.sqrt(np.maximum(np.asarray(cluster.variance), variance_floor))
    return float(np.sqrt(np.mean((difference / scale) ** 2)))


def add_target(
    state: TrainingState,
    target: StoredTarget,
    config: ContextConfig | None = None,
) -> str:
    """Add one target, update its online context cluster, and rebalance state."""
    policy = config or ContextConfig()
    if any(item.sequence == target.sequence for item in state.targets):
        msg = "training target sequence already exists"
        raise ValueError(msg)
    compatible = [
        cluster
        for cluster in state.clusters
        if cluster.camera_id == target.camera_id and cluster.feature_schema == target.feature_schema
    ]
    nearest = min(
        compatible,
        key=lambda cluster: (
            context_distance(target.context, cluster, policy.variance_floor),
            cluster.cluster_id,
        ),
        default=None,
    )
    if (
        nearest is None
        or context_distance(target.context, nearest, policy.variance_floor)
        > policy.assignment_distance
    ):
        if len(compatible) >= policy.maximum_clusters_per_partition:
            _make_cluster_room(state, compatible, policy)
        cluster_id = f"context-{target.sequence}"
        cluster = ContextCluster(
            cluster_id,
            target.camera_id,
            target.feature_schema,
            target.context,
            tuple(policy.variance_floor for _ in target.context),
            1,
            (target.sequence,),
        )
        state.clusters.append(cluster)
    else:
        cluster_id = nearest.cluster_id

    state.targets.append(target)
    state.next_sequence = max(state.next_sequence, target.sequence + 1)
    _rebalance(state, cluster_id, policy)
    return cluster_id


def balanced_coreset(
    targets: Sequence[StoredTarget],
    limit: int,
) -> list[StoredTarget]:
    """Retain deterministic output/zone balance and farthest-point diversity."""
    if limit <= 0:
        msg = "coreset limit must be positive"
        raise ValueError(msg)
    buckets: dict[tuple[str, str, str, str], list[StoredTarget]] = {}
    for target in targets:
        key = (target.camera_id, target.feature_schema, target.output_key, target.zone)
        buckets.setdefault(key, []).append(target)
    ranked = {key: _farthest_order(items) for key, items in sorted(buckets.items())}
    selected: list[StoredTarget] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for key in sorted(ranked):
            items = ranked[key]
            if depth < len(items):
                selected.append(items[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return sorted(selected, key=lambda item: item.sequence)


def calibration_samples_for(
    state: TrainingState,
    topology: DisplayTopology,
    *,
    camera_id: str,
    feature_schema: str,
) -> list[CalibrationSample]:
    """Return compatible stored targets remapped to current geometry."""
    compatible = [
        target
        for target in state.targets
        if target.camera_id == camera_id and target.feature_schema == feature_schema
    ]
    return [sample for _sequence, sample, _quality in _mapped_samples(compatible, topology)]


def build_router(
    state: TrainingState,
    topology: DisplayTopology,
    *,
    camera_id: str,
    feature_schema: str,
    config: ContextConfig | None = None,
) -> ModelRouter:
    """Refit compatible global and local estimators for current geometry."""
    policy = config or ContextConfig()
    compatible = [
        target
        for target in state.targets
        if target.camera_id == camera_id and target.feature_schema == feature_schema
    ]
    mapped = _mapped_samples(compatible, topology)
    if len(mapped) < MINIMUM_MODEL_TARGETS:
        msg = "stored training data do not contain three compatible targets"
        raise ValueError(msg)
    quality = min(item[2] for item in mapped)
    prefix = f"{camera_id}:{topology.topology_id}:"
    stored_global = state.models.get(f"{prefix}global")
    if stored_global is not None:
        try:
            global_model = CalibrationModel.from_record(stored_global)
            exact_experts = [
                ContextExpert(
                    cluster,
                    CalibrationModel.from_record(state.models[f"{prefix}{cluster.cluster_id}"]),
                )
                for cluster in state.clusters
                if cluster.camera_id == camera_id
                and cluster.feature_schema == feature_schema
                and f"{prefix}{cluster.cluster_id}" in state.models
            ]
            return ModelRouter(
                global_model,
                exact_experts,
                camera_id=camera_id,
                feature_schema=feature_schema,
                topology_quality=TopologyQuality.EXACT,
                config=policy,
            )
        except (KeyError, TypeError, ValueError):
            pass
    global_model = CalibrationModel.fit([sample for _sequence, sample, _quality in mapped])
    by_sequence = {sequence: sample for sequence, sample, _item_quality in mapped}
    experts: list[ContextExpert] = []
    for cluster in state.clusters:
        if cluster.camera_id != camera_id or cluster.feature_schema != feature_schema:
            continue
        samples = [
            by_sequence[sequence]
            for sequence in cluster.target_sequences
            if sequence in by_sequence
        ]
        if len(samples) >= MINIMUM_MODEL_TARGETS:
            experts.append(ContextExpert(cluster, CalibrationModel.fit(samples)))
    return ModelRouter(
        global_model,
        experts,
        camera_id=camera_id,
        feature_schema=feature_schema,
        topology_quality=quality,
        config=policy,
    )


def candidate_is_acceptable(
    incumbent: ValidationMetrics | None,
    candidate: ValidationMetrics,
    *,
    tolerance: float = 0.0,
) -> bool:
    """Accept new contexts or candidates that do not regress either quality gate."""
    if tolerance < 0.0:
        msg = "acceptance tolerance must be non-negative"
        raise ValueError(msg)
    if incumbent is None:
        return True
    return (
        candidate.median_error <= incumbent.median_error + tolerance
        and candidate.edge_error <= incumbent.edge_error + tolerance
    )


def _mapped_samples(
    targets: Iterable[StoredTarget],
    topology: DisplayTopology,
) -> list[tuple[int, CalibrationSample, TopologyQuality]]:
    result: list[tuple[int, CalibrationSample, TopologyQuality]] = []
    for target in targets:
        mapped = map_stored_target(target, topology)
        if mapped is not None:
            result.append(
                (
                    target.sequence,
                    CalibrationSample(target.features, mapped.point),
                    mapped.quality,
                )
            )
    return result


def _make_cluster_room(
    state: TrainingState,
    compatible: list[ContextCluster],
    config: ContextConfig,
) -> None:
    pairs = [
        (
            context_distance(left.centroid, right, config.variance_floor),
            left.cluster_id,
            right.cluster_id,
        )
        for index, left in enumerate(compatible)
        for right in compatible[index + 1 :]
    ]
    if pairs:
        distance, left_id, right_id = min(pairs)
        if distance <= config.merge_distance:
            sequences = next(
                cluster.target_sequences for cluster in compatible if cluster.cluster_id == left_id
            ) + next(
                cluster.target_sequences for cluster in compatible if cluster.cluster_id == right_id
            )
            state.clusters = [
                cluster
                for cluster in state.clusters
                if cluster.cluster_id not in {left_id, right_id}
            ]
            samples = [target for target in state.targets if target.sequence in sequences]
            if samples:
                state.clusters.append(_cluster_from_targets(left_id, samples, config))
            return
    evicted = min(
        compatible, key=lambda cluster: (len(cluster.target_sequences), cluster.cluster_id)
    )
    state.clusters = [
        cluster for cluster in state.clusters if cluster.cluster_id != evicted.cluster_id
    ]


def _rebalance(state: TrainingState, cluster_id: str, config: ContextConfig) -> None:
    state.targets = balanced_coreset(state.targets, config.maximum_global_targets)
    available = {target.sequence: target for target in state.targets}
    rebuilt: list[ContextCluster] = []
    for cluster in state.clusters:
        sequences = list(cluster.target_sequences)
        if cluster.cluster_id == cluster_id and state.targets:
            newest = max(state.targets, key=lambda item: item.sequence)
            if newest.sequence not in sequences:
                sequences.append(newest.sequence)
        samples = [available[sequence] for sequence in sequences if sequence in available]
        samples = balanced_coreset(samples, config.maximum_cluster_targets) if samples else []
        if samples:
            rebuilt.append(_cluster_from_targets(cluster.cluster_id, samples, config, cluster))
    state.clusters = rebuilt


def _cluster_from_targets(
    cluster_id: str,
    targets: Sequence[StoredTarget],
    config: ContextConfig,
    prior: ContextCluster | None = None,
) -> ContextCluster:
    contexts = np.asarray([target.context for target in targets], dtype=np.float64)
    centroid = contexts.mean(axis=0)
    variance = np.maximum(contexts.var(axis=0), config.variance_floor)
    first = targets[0]
    return ContextCluster(
        cluster_id,
        first.camera_id,
        first.feature_schema,
        tuple(float(value) for value in centroid),
        tuple(float(value) for value in variance),
        (prior.sample_count if prior is not None else 0) + 1,
        tuple(target.sequence for target in targets),
        median_error=None if prior is None else prior.median_error,
        edge_error=None if prior is None else prior.edge_error,
    )


def _farthest_order(targets: Sequence[StoredTarget]) -> list[StoredTarget]:
    remaining = sorted(targets, key=lambda item: item.sequence)
    if not remaining:
        return []
    selected = [remaining.pop(0)]
    while remaining:
        candidate = max(
            remaining,
            key=lambda item: (
                min(_target_distance(item, prior) for prior in selected),
                -item.sequence,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def _target_distance(left: StoredTarget, right: StoredTarget) -> float:
    left_vector = np.asarray((*left.context, *left.features), dtype=np.float64)
    right_vector = np.asarray((*right.context, *right.features), dtype=np.float64)
    if left_vector.shape != right_vector.shape:
        return math.inf
    return float(np.linalg.norm(left_vector - right_vector))


def _validation_weight(cluster: ContextCluster) -> float:
    errors = [value for value in (cluster.median_error, cluster.edge_error) if value is not None]
    if not errors:
        return 1.0
    return 1.0 / (1.0 + max(errors) / 100.0)
