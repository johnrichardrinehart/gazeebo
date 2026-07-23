"""Tests for implicit context clustering and model routing."""

from __future__ import annotations

import unittest

from gazeebo.adaptation import TopologyQuality
from gazeebo.calibration import CalibrationModel, CalibrationSample
from gazeebo.contexts import (
    ContextConfig,
    ContextExpert,
    ModelRouter,
    ValidationMetrics,
    add_target,
    balanced_coreset,
    candidate_is_acceptable,
)
from gazeebo.geometry import Point
from gazeebo.state import ContextCluster, OutputDescriptor, StoredTarget, TrainingState


def target(
    sequence: int,
    context: tuple[float, ...],
    *,
    output: str = "display",
    zone: str = "center",
) -> StoredTarget:
    """Create one target for clustering and retention tests."""
    return StoredTarget(
        sequence=sequence,
        camera_id="camera-a",
        feature_schema="gaze-v1",
        features=(sequence / 10.0, context[0]),
        context=context,
        outputs=(OutputDescriptor(output, 0, 0, 1000, 700),),
        output_key=output,
        target_u=0.5,
        target_v=0.5,
        desktop_u=0.5,
        desktop_v=0.5,
        zone=zone,
    )


def model(offset: float) -> CalibrationModel:
    """Fit a deterministic one-dimensional affine estimator."""
    return CalibrationModel.fit(
        (
            CalibrationSample((0.0,), Point(offset, 0.0)),
            CalibrationSample((0.5,), Point(offset + 50.0, 0.0)),
            CalibrationSample((1.0,), Point(offset + 100.0, 0.0)),
        )
    )


def cluster(name: str, center: float, error: float = 50.0) -> ContextCluster:
    """Create one validated routing context."""
    return ContextCluster(
        name,
        "camera-a",
        "gaze-v1",
        (center, 0.5),
        (0.01, 0.01),
        3,
        (0, 1, 2),
        median_error=error,
        edge_error=error,
    )


class ContextTests(unittest.TestCase):
    """Lock automatic clusters, bounded coresets, routing, and acceptance."""

    def test_online_assignment_groups_near_contexts_and_splits_far_contexts(self) -> None:
        """Posture and illumination neighborhoods emerge without profile names."""
        state = TrainingState()
        config = ContextConfig(variance_floor=0.01)
        first = add_target(state, target(0, (0.0, 0.5)), config)
        nearby = add_target(state, target(1, (0.05, 0.52)), config)
        distant = add_target(state, target(2, (1.0, 0.1)), config)
        assert first == nearby
        assert distant != first
        assert len(state.clusters) == 2
        assert sorted(len(item.target_sequences) for item in state.clusters) == [1, 2]

    def test_cluster_count_and_membership_are_bounded(self) -> None:
        """Novel contexts merge or evict rather than growing without limit."""
        state = TrainingState()
        config = ContextConfig(
            maximum_clusters_per_partition=2,
            maximum_cluster_targets=2,
            maximum_global_targets=5,
            assignment_distance=0.5,
            merge_distance=0.1,
            variance_floor=0.01,
        )
        for index, center in enumerate((0.0, 1.0, 2.0, 2.1, 2.2, 2.3)):
            add_target(state, target(index, (center, center)), config)
        assert len(state.clusters) <= 2
        assert len(state.targets) <= 5
        assert all(len(item.target_sequences) <= 2 for item in state.clusters)

    def test_balanced_coreset_retains_sparse_output_and_corner_evidence(self) -> None:
        """Repeated center training cannot erase a sparse output/zone bucket."""
        common = [target(index, (index / 100.0, 0.5)) for index in range(10)]
        sparse = target(10, (1.0, 0.1), output="other", zone="corner")
        selected = balanced_coreset((*common, sparse), 4)
        assert sparse in selected
        assert {item.output_key for item in selected} == {"display", "other"}
        assert {item.zone for item in selected} == {"center", "corner"}

    def test_router_selects_posture_experts_and_falls_back_out_of_distribution(self) -> None:
        """Passive context chooses local experts without exposing profile controls."""
        router = ModelRouter(
            model(400.0),
            (
                ContextExpert(cluster("seated", 0.0), model(100.0)),
                ContextExpert(cluster("standing", 1.0), model(900.0)),
            ),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.EXACT,
            config=ContextConfig(variance_floor=0.01, routing_smoothing=0.5),
        )
        seated_point, seated = router.predict_with_decision((0.5,), (0.0, 0.5))
        assert "seated" in seated.label
        assert seated.confidence_label == "inferred-compatible"
        assert seated_point.x < 500.0

        standing_point = seated_point
        standing = seated
        for _ in range(8):
            standing_point, standing = router.predict_with_decision((0.5,), (1.0, 0.5))
        assert "standing" in standing.label
        assert standing_point.x > 500.0

        fallback_point, fallback = router.predict_with_decision((0.5,), (10.0, 10.0))
        assert fallback.out_of_distribution
        assert fallback.confidence_label == "inferred-low"
        assert fallback.weights[0][0] == "global"
        assert 0.0 <= fallback_point.x <= 1000.0

    def test_weak_topology_confidence_remains_explicitly_inferred(self) -> None:
        """A mapped model cannot present topology compatibility as holdout accuracy."""
        router = ModelRouter(
            model(400.0),
            (ContextExpert(cluster("near", 0.0), model(100.0)),),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.WEAK,
        )
        _point, decision = router.predict_with_decision((0.5,), (0.0, 0.5))
        assert decision.confidence_label == "inferred-weak"

    def test_routing_weights_change_smoothly(self) -> None:
        """A small context boundary crossing cannot instantly replace all weight."""
        router = ModelRouter(
            model(400.0),
            (
                ContextExpert(cluster("left-context", 0.0), model(100.0)),
                ContextExpert(cluster("right-context", 0.2), model(900.0)),
            ),
            camera_id="camera-a",
            feature_schema="gaze-v1",
            topology_quality=TopologyQuality.STRONG,
            config=ContextConfig(variance_floor=0.01, routing_smoothing=0.1, switching_margin=0.2),
        )
        _point, before = router.predict_with_decision((0.5,), (0.0, 0.5))
        _point, after = router.predict_with_decision((0.5,), (0.2, 0.5))
        before_weights = dict(before.weights)
        after_weights = dict(after.weights)
        assert after_weights.get("left-context", 0.0) > 0.0
        assert (
            abs(after_weights.get("right-context", 0.0) - before_weights.get("right-context", 0.0))
            < 0.2
        )

    def test_candidate_must_not_regress_either_accuracy_gate(self) -> None:
        """Persistent updates preserve both median and edge/corner quality."""
        incumbent = ValidationMetrics(90.0, 95.0)
        assert candidate_is_acceptable(incumbent, ValidationMetrics(80.0, 95.0))
        assert not candidate_is_acceptable(incumbent, ValidationMetrics(80.0, 96.0))
        assert not candidate_is_acceptable(incumbent, ValidationMetrics(91.0, 90.0))
        assert candidate_is_acceptable(None, ValidationMetrics(500.0, 600.0))

    def test_model_coefficients_round_trip_without_training_samples(self) -> None:
        """The store can restore fitted coefficients after restart."""
        original = model(123.0)
        restored = CalibrationModel.from_record(original.to_record())
        expected = original.predict((0.25,))
        actual = restored.predict((0.25,))
        assert abs(actual.x - expected.x) < 1e-9
        assert abs(actual.y - expected.y) < 1e-9


if __name__ == "__main__":
    unittest.main()
