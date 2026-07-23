"""Ephemeral gaze calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gazeebo.geometry import Point

MINIMUM_CALIBRATION_SAMPLES = 3
MINIMUM_MODEL_SELECTION_SAMPLES = 8
BASE_GAZE_FEATURE_COUNT = 8
BINOCULAR_GAZE_FEATURE_COUNT = 10
MINIMUM_FEATURE_SCALE = 1e-6


def _record_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError
    return int(value)


def _record_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError
    return float(value)


def _record_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError
    return value


if TYPE_CHECKING:
    from collections.abc import Sequence

    from gazeebo.contracts import FeatureVector


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One gaze feature vector paired with a visible global target."""

    features: FeatureVector
    target: Point


def aggregate_features(vectors: Sequence[FeatureVector]) -> FeatureVector:
    """Reduce one target's frame samples to a robust feature vector."""
    if not vectors:
        msg = "at least one feature vector is required"
        raise ValueError(msg)
    feature_count = len(vectors[0])
    if feature_count == 0 or any(len(vector) != feature_count for vector in vectors):
        msg = "feature vectors must have one consistent non-zero length"
        raise ValueError(msg)
    matrix = np.asarray(vectors, dtype=np.float64)
    return tuple(float(value) for value in np.median(matrix, axis=0))


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    ridge: float
    feature_indices: tuple[int, ...]
    feature_name: str
    gamma: float = 0.0


class CalibrationModel:
    """Map gaze features to coordinates with a validated session estimator."""

    def __init__(  # noqa: PLR0913
        self,
        coefficients: np.ndarray,
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        *,
        kind: str = "affine",
        support: np.ndarray | None = None,
        gamma: float = 0.0,
        target_offset: np.ndarray | None = None,
        input_feature_count: int | None = None,
        feature_indices: tuple[int, ...] | None = None,
        feature_name: str = "all",
    ) -> None:
        """Store one fitted, process-local estimator."""
        self._coefficients = coefficients
        self._feature_mean = feature_mean
        self._feature_scale = feature_scale
        self._input_feature_count = (
            len(feature_mean) if input_feature_count is None else input_feature_count
        )
        self._feature_indices = (
            tuple(range(len(feature_mean))) if feature_indices is None else feature_indices
        )
        self._kind = kind
        self._feature_name = feature_name
        self._support = support
        self._gamma = gamma
        self._target_offset = (
            np.zeros(2, dtype=np.float64) if target_offset is None else target_offset
        )

    @property
    def kind(self) -> str:
        """Name the estimator selected from session calibration targets."""
        if self._feature_name == "all":
            return self._kind
        return f"{self._kind}/{self._feature_name}"

    @classmethod
    def fit(
        cls,
        samples: Sequence[CalibrationSample],
        *,
        ridge: float = 0.1,
    ) -> CalibrationModel:
        """Select and fit an estimator using leave-one-target-out error."""
        if ridge < 0.0:
            msg = "ridge regularization must be non-negative"
            raise ValueError(msg)
        if len(samples) < MINIMUM_CALIBRATION_SAMPLES:
            msg = "at least three calibration samples are required"
            raise ValueError(msg)
        feature_count = len(samples[0].features)
        if feature_count == 0 or any(len(sample.features) != feature_count for sample in samples):
            msg = "calibration feature vectors must have one consistent non-zero length"
            raise ValueError(msg)

        features = np.asarray([sample.features for sample in samples], dtype=np.float64)
        targets = np.asarray(
            [(sample.target.x, sample.target.y) for sample in samples],
            dtype=np.float64,
        )
        all_features = tuple(range(feature_count))
        feature_sets: tuple[tuple[tuple[int, ...], str], ...] = ((all_features, "all"),)
        if feature_count >= BINOCULAR_GAZE_FEATURE_COUNT:
            feature_sets = (
                (all_features, "all"),
                ((8, 9), "binocular"),
                ((4, 5, 8, 9), "binocular+pose"),
                ((6, 7, 8, 9), "binocular+position"),
                ((4, 5, 6, 7, 8, 9), "binocular+context"),
                ((0, 1, 2, 3), "pupils"),
                ((0, 1, 2, 3, 4, 5), "pupils+pose"),
                ((0, 1, 2, 3, 6, 7), "pupils+position"),
            )
        elif feature_count >= BASE_GAZE_FEATURE_COUNT:
            feature_sets = (
                (all_features, "all"),
                ((0, 1, 2, 3), "pupils"),
                ((0, 1, 2, 3, 4, 5), "pupils+pose"),
                ((0, 1, 2, 3, 6, 7), "pupils+position"),
            )
        candidates: tuple[_Candidate, ...] = (_Candidate("affine", ridge, all_features, "all"),)
        if len(samples) >= MINIMUM_MODEL_SELECTION_SAMPLES:
            regularization = tuple(
                sorted({max(ridge * factor, 1e-10) for factor in (0.1, 1.0, 10.0)})
            )
            candidates = tuple(
                [
                    _Candidate("affine", value, indices, name)
                    for indices, name in feature_sets
                    for value in regularization
                ]
                + [
                    _Candidate("rbf", value, indices, name, gamma)
                    for indices, name in feature_sets
                    for gamma in (0.05, 0.1, 0.25, 0.5)
                    for value in regularization
                ]
            )
            candidate = min(
                candidates,
                key=lambda option: cls._cross_validation_error(
                    features,
                    targets,
                    option,
                ),
            )
        else:
            candidate = candidates[0]
        return cls._fit_candidate(features, targets, candidate)

    @classmethod
    def _cross_validation_error(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        candidate: _Candidate,
    ) -> float:
        """Score one candidate only on targets excluded from its fit."""
        errors: list[float] = []
        for held_out in range(len(features)):
            mask = np.arange(len(features)) != held_out
            model = cls._fit_candidate(features[mask], targets[mask], candidate)
            prediction = model.predict(tuple(float(value) for value in features[held_out]))
            error = np.hypot(
                prediction.x - targets[held_out, 0],
                prediction.y - targets[held_out, 1],
            )
            errors.append(float(error))
        values = np.asarray(errors, dtype=np.float64)
        return float(np.median(values) + 0.25 * np.percentile(values, 90.0))

    @classmethod
    def _fit_candidate(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        candidate: _Candidate,
    ) -> CalibrationModel:
        """Fit one candidate to all supplied target anchors."""
        selected = features[:, candidate.feature_indices]
        feature_mean = selected.mean(axis=0)
        feature_scale = selected.std(axis=0)
        feature_scale = np.where(feature_scale < MINIMUM_FEATURE_SCALE, 1.0, feature_scale)
        normalized = (selected - feature_mean) / feature_scale
        if candidate.kind == "rbf":
            distances = np.sum(
                (normalized[:, np.newaxis, :] - normalized[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            kernel = np.exp(-candidate.gamma * distances)
            target_offset = targets.mean(axis=0)
            coefficients = np.linalg.solve(
                kernel + np.eye(len(features), dtype=np.float64) * candidate.ridge,
                targets - target_offset,
            )
            return cls(
                coefficients,
                feature_mean,
                feature_scale,
                kind="rbf",
                support=normalized,
                gamma=candidate.gamma,
                target_offset=target_offset,
                input_feature_count=features.shape[1],
                feature_indices=candidate.feature_indices,
                feature_name=candidate.feature_name,
            )

        design = np.column_stack((np.ones(len(features)), normalized))
        penalty = np.eye(design.shape[1], dtype=np.float64) * candidate.ridge
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        return cls(
            coefficients,
            feature_mean,
            feature_scale,
            input_feature_count=features.shape[1],
            feature_indices=candidate.feature_indices,
            feature_name=candidate.feature_name,
        )

    def to_record(self) -> dict[str, object]:
        """Serialize fitted coefficients without retaining training observations."""
        return {
            "kind": self._kind,
            "coefficients": self._coefficients.tolist(),
            "feature_mean": self._feature_mean.tolist(),
            "feature_scale": self._feature_scale.tolist(),
            "support": None if self._support is None else self._support.tolist(),
            "gamma": self._gamma,
            "target_offset": self._target_offset.tolist(),
            "input_feature_count": self._input_feature_count,
            "feature_indices": list(self._feature_indices),
            "feature_name": self._feature_name,
        }

    @classmethod
    def from_record(cls, record: dict[str, object]) -> CalibrationModel:
        """Restore validated fitted coefficients from the local training store."""
        try:
            kind = str(record["kind"])
            coefficients = np.asarray(record["coefficients"], dtype=np.float64)
            feature_mean = np.asarray(record["feature_mean"], dtype=np.float64)
            feature_scale = np.asarray(record["feature_scale"], dtype=np.float64)
            raw_support = record.get("support")
            support = None if raw_support is None else np.asarray(raw_support, dtype=np.float64)
            gamma = _record_number(record["gamma"])
            target_offset = np.asarray(record["target_offset"], dtype=np.float64)
            input_feature_count = _record_integer(record["input_feature_count"])
            raw_indices = _record_list(record["feature_indices"])
            feature_indices = tuple(_record_integer(value) for value in raw_indices)
            feature_name = str(record["feature_name"])
        except (KeyError, TypeError, ValueError) as error:
            msg = "stored calibration model is malformed"
            raise ValueError(msg) from error
        arrays = [coefficients, feature_mean, feature_scale, target_offset]
        if support is not None:
            arrays.append(support)
        if (
            kind not in {"affine", "rbf"}
            or not feature_indices
            or input_feature_count <= 0
            or any(not np.all(np.isfinite(array)) for array in arrays)
            or np.any(feature_scale <= 0.0)
        ):
            msg = "stored calibration model is invalid"
            raise ValueError(msg)
        return cls(
            coefficients,
            feature_mean,
            feature_scale,
            kind=kind,
            support=support,
            gamma=gamma,
            target_offset=target_offset,
            input_feature_count=input_feature_count,
            feature_indices=feature_indices,
            feature_name=feature_name,
        )

    def predict(
        self,
        features: FeatureVector,
        context: tuple[float, ...] | None = None,
    ) -> Point:
        """Predict one global point; affine calibration does not route on context."""
        del context
        if len(features) != self._input_feature_count:
            msg = "gaze feature vector does not match calibration"
            raise ValueError(msg)
        selected = np.asarray(features, dtype=np.float64)[list(self._feature_indices)]
        normalized = (selected - self._feature_mean) / self._feature_scale
        if self._kind == "rbf":
            if self._support is None:
                msg = "RBF calibration support is unavailable"
                raise RuntimeError(msg)
            distances = np.sum((self._support - normalized) ** 2, axis=1)
            row = np.exp(-self._gamma * distances)
            prediction = self._target_offset + row @ self._coefficients
        else:
            row = np.concatenate(([1.0], normalized))
            prediction = row @ self._coefficients
        return Point(float(prediction[0]), float(prediction[1]))
