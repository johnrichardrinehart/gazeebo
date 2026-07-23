"""Tests for authorized-display calibration, clipping, and smoothing."""

from __future__ import annotations

import unittest

from gazeebo.calibration import (
    CalibrationModel,
    CalibrationSample,
    aggregate_features,
)
from gazeebo.contracts import DisplayRegion
from gazeebo.geometry import (
    DisplayTopology,
    Point,
    PointerSmoother,
    PointerTarget,
    calibration_targets,
)


class GeometryTests(unittest.TestCase):
    """Lock full-desktop mapping without a graphical session."""

    def setUp(self) -> None:
        """Create one representative authorized display."""
        self.topology = DisplayTopology((DisplayRegion("selected", 3840, 360, 2560, 1440),))

    def test_targets_cover_the_selected_display(self) -> None:
        """Calibration contributes center and corner samples to one display."""
        targets = calibration_targets(self.topology)
        assert len(targets) == 5
        assert {target.region_id for target in targets} == {"selected"}
        for target in targets:
            global_point = self.topology.to_global(target)
            located = self.topology.locate(global_point)
            self.assertAlmostEqual(located.x, target.x)
            self.assertAlmostEqual(located.y, target.y)

    def test_calibration_targets_cover_every_authorized_display(self) -> None:
        """Combined calibration contributes five anchors per display."""
        topology = DisplayTopology(
            (
                DisplayRegion("first", 0, 0, 1000, 700),
                DisplayRegion("second", 1000, 100, 800, 600),
            )
        )
        targets = calibration_targets(topology)
        assert len(targets) == 10
        assert [target.region_id for target in targets].count("first") == 5
        assert [target.region_id for target in targets].count("second") == 5

    def test_predictions_clip_to_selected_display_edges(self) -> None:
        """No fitted point can roam onto another output."""
        upper_left = self.topology.locate(Point(-100.0, -100.0))
        lower_right = self.topology.locate(Point(10000.0, 10000.0))
        assert upper_left == PointerTarget("selected", 0.0, 0.0)
        assert lower_right.region_id == "selected"
        assert 2559.0 < lower_right.x < 2560.0
        assert 1439.0 < lower_right.y < 1440.0

    def test_global_and_local_mapping_retains_selected_origin(self) -> None:
        """Global diagnostics retain an authorized display's compositor origin."""
        local = PointerTarget("selected", 100.0, 200.0)
        assert self.topology.to_global(local) == Point(3940.0, 560.0)
        assert self.topology.locate(Point(3940.0, 560.0)) == local

    def test_topology_identity_changes_with_selected_geometry(self) -> None:
        """Stale calibration detects movement or resizing of authorized displays."""
        changed = DisplayTopology((DisplayRegion("selected", 0, 0, 1920, 1080),))
        same = DisplayTopology(self.topology.regions)
        assert same.topology_id == self.topology.topology_id
        assert changed.topology_id != self.topology.topology_id

    def test_multiple_regions_allow_roaming_and_clip_union_gaps(self) -> None:
        """Predictions cross displays but cannot remain in unauthorized gaps."""
        topology = DisplayTopology(
            (
                DisplayRegion("first", 0, 0, 1000, 700),
                DisplayRegion("second", 1200, 100, 500, 500),
            )
        )
        assert topology.locate(Point(1400.0, 300.0)) == PointerTarget(
            "second",
            200.0,
            200.0,
        )
        gap = topology.locate(Point(1080.0, 300.0))
        assert gap.region_id == "first"
        assert 999.0 < gap.x < 1000.0
        assert topology.to_global(gap).x < 1000.0

    def test_duplicate_region_ids_are_rejected(self) -> None:
        """Pointer stream lookup stays unambiguous across displays."""
        with self.assertRaisesRegex(ValueError, "unique"):
            DisplayTopology(
                (
                    DisplayRegion("same", 0, 0, 1000, 700),
                    DisplayRegion("same", 1000, 0, 1000, 700),
                )
            )

    def test_target_features_use_robust_component_medians(self) -> None:
        """Frame jitter cannot attenuate calibration response within one target."""
        assert aggregate_features(((0.1, 0.8), (0.2, 0.7), (9.0, 0.6), (0.3, 0.5), (0.4, 0.4))) == (
            0.3,
            0.6,
        )

    def test_affine_calibration_recovers_known_mapping(self) -> None:
        """Ridge fitting maps gaze features into global logical coordinates."""
        samples = []
        for first, second in (
            (-1.0, -1.0),
            (-1.0, 0.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, -1.0),
            (1.0, 0.0),
            (1.0, 1.0),
        ):
            samples.append(
                CalibrationSample(
                    (first, second),
                    Point(100.0 + 500.0 * first, 200.0 + 300.0 * second),
                )
            )
        model = CalibrationModel.fit(samples, ridge=1e-10)
        prediction = model.predict((0.25, -0.5))
        assert model.kind == "affine"
        self.assertAlmostEqual(prediction.x, 225.0, places=5)
        self.assertAlmostEqual(prediction.y, 50.0, places=5)

    def test_cross_validation_selects_nonlinear_estimator_when_supported(self) -> None:
        """Target holdouts select RBF interpolation for a nonlinear response."""
        samples = [
            CalibrationSample(
                (value,),
                Point(1000.0 + 800.0 * value**3, 500.0 + 400.0 * value**2),
            )
            for value in (-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        model = CalibrationModel.fit(samples)
        prediction = model.predict((0.5,))
        assert model.kind == "rbf"
        self.assertLess(abs(prediction.x - 1100.0), 80.0)
        self.assertLess(abs(prediction.y - 600.0), 80.0)

    def test_smoothing_suppresses_jitter_and_caps_repositioning(self) -> None:
        """Small idle changes stop while large gaze changes pan in bounded steps."""
        smoother = PointerSmoother(alpha=0.5, dead_zone=5.0, maximum_step=100.0)
        assert smoother.update(Point(10.0, 10.0)) == Point(10.0, 10.0)
        assert smoother.update(Point(13.0, 13.0)) == Point(10.0, 10.0)
        assert smoother.update(Point(1010.0, 10.0)) == Point(110.0, 10.0)
        assert smoother.update(Point(210.0, 10.0)) == Point(160.0, 10.0)
        smoother.reset()
        assert smoother.update(Point(-50.0, 20.0)) == Point(-50.0, 20.0)


if __name__ == "__main__":
    unittest.main()
