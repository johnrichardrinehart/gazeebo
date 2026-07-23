"""Secure, bounded persistence for target-level training data."""

from __future__ import annotations

import contextlib
import json
import math
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

STORE_VERSION = 1
MAXIMUM_STORED_TARGETS = 256
MAXIMUM_STORED_CLUSTERS = 64
MAXIMUM_VALIDATION_RECORDS = 64
_DIRECTORY_MODE = 0o700
_FILE_MODE = 0o600


class TrainingStoreError(RuntimeError):
    """The local training store is unsafe or malformed."""


@dataclass(frozen=True, slots=True)
class OutputDescriptor:
    """Generic logical identity and geometry for one authorized output."""

    key: str
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject unusable stored output geometry."""
        if not self.key or self.width <= 0 or self.height <= 0:
            msg = "stored output descriptor is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StoredTarget:
    """One target-level aggregate without its source frame stream."""

    sequence: int
    camera_id: str
    feature_schema: str
    features: tuple[float, ...]
    context: tuple[float, ...]
    outputs: tuple[OutputDescriptor, ...]
    output_key: str
    target_u: float
    target_v: float
    desktop_u: float
    desktop_v: float
    zone: str

    def __post_init__(self) -> None:
        """Reject records that cannot be routed or remapped safely."""
        values = (
            *self.features,
            *self.context,
            self.target_u,
            self.target_v,
            self.desktop_u,
            self.desktop_v,
        )
        if self.sequence < 0 or not self.camera_id or not self.feature_schema:
            msg = "stored target identity is invalid"
            raise ValueError(msg)
        if (
            not self.features
            or not self.context
            or not all(math.isfinite(value) for value in values)
        ):
            msg = "stored target vectors must contain finite values"
            raise ValueError(msg)
        if not self.outputs or self.output_key not in {output.key for output in self.outputs}:
            msg = "stored target output is unavailable"
            raise ValueError(msg)
        if not all(
            0.0 <= value <= 1.0
            for value in (self.target_u, self.target_v, self.desktop_u, self.desktop_v)
        ):
            msg = "stored target coordinates must be normalized"
            raise ValueError(msg)
        if self.zone not in {"center", "edge", "corner"}:
            msg = "stored target zone is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ContextCluster:
    """Bounded routing statistics and target membership for one context."""

    cluster_id: str
    camera_id: str
    feature_schema: str
    centroid: tuple[float, ...]
    variance: tuple[float, ...]
    sample_count: int
    target_sequences: tuple[int, ...]
    median_error: float | None = None
    edge_error: float | None = None

    def __post_init__(self) -> None:
        """Reject malformed bounded-cluster statistics."""
        if not self.cluster_id or not self.camera_id or not self.feature_schema:
            msg = "context cluster identity is invalid"
            raise ValueError(msg)
        if not self.centroid or len(self.centroid) != len(self.variance):
            msg = "context cluster dimensions are invalid"
            raise ValueError(msg)
        values = (*self.centroid, *self.variance)
        if not all(math.isfinite(value) for value in values) or any(
            value < 0.0 for value in self.variance
        ):
            msg = "context cluster statistics are invalid"
            raise ValueError(msg)
        if self.sample_count <= 0 or any(sequence < 0 for sequence in self.target_sequences):
            msg = "context cluster membership is invalid"
            raise ValueError(msg)
        errors = (self.median_error, self.edge_error)
        if any(value is not None and (not math.isfinite(value) or value < 0.0) for value in errors):
            msg = "context cluster validation is invalid"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Aggregate unseen quality without retaining its observations."""

    sequence: int
    camera_id: str
    topology_id: str
    routing: str
    median_error: float
    edge_error: float

    def __post_init__(self) -> None:
        """Reject malformed aggregate holdout metrics."""
        if self.sequence < 0 or not self.camera_id or not self.topology_id or not self.routing:
            msg = "validation identity is invalid"
            raise ValueError(msg)
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.median_error, self.edge_error)
        ):
            msg = "validation errors are invalid"
            raise ValueError(msg)


@dataclass(slots=True)
class TrainingState:
    """Complete versioned state replaced by one atomic transaction."""

    next_sequence: int = 0
    targets: list[StoredTarget] = field(default_factory=list)
    clusters: list[ContextCluster] = field(default_factory=list)
    models: dict[str, dict[str, object]] = field(default_factory=dict)
    validations: list[ValidationSummary] = field(default_factory=list)

    def validate(self) -> None:
        """Reject unbounded or internally inconsistent state."""
        if self.next_sequence < 0:
            msg = "training sequence is invalid"
            raise TrainingStoreError(msg)
        if len(self.targets) > MAXIMUM_STORED_TARGETS:
            msg = "training store exceeds its target limit"
            raise TrainingStoreError(msg)
        if len(self.clusters) > MAXIMUM_STORED_CLUSTERS:
            msg = "training store exceeds its cluster limit"
            raise TrainingStoreError(msg)
        if len(self.validations) > MAXIMUM_VALIDATION_RECORDS:
            msg = "training store exceeds its validation limit"
            raise TrainingStoreError(msg)
        sequences = [target.sequence for target in self.targets]
        if len(sequences) != len(set(sequences)):
            msg = "training target sequences must be unique"
            raise TrainingStoreError(msg)
        available = set(sequences)
        if any(not set(cluster.target_sequences) <= available for cluster in self.clusters):
            msg = "context cluster refers to a missing target"
            raise TrainingStoreError(msg)
        if any(
            not isinstance(key, str) or not isinstance(value, dict)
            for key, value in self.models.items()
        ):
            msg = "stored model map is malformed"
            raise TrainingStoreError(msg)


class TrainingStore:
    """Read and atomically replace one owner-only local training store."""

    def __init__(self, path: Path | None = None, *, ephemeral: bool = False) -> None:
        """Use the XDG store or an injected deterministic test path."""
        self.path = path or _default_path()
        self.ephemeral = ephemeral

    def load(self) -> TrainingState:
        """Load validated state, or empty state when unavailable or ephemeral."""
        if self.ephemeral or not self.path.exists():
            return TrainingState()
        self._validate_directory(create=False)
        self._validate_file()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return _decode_state(raw)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            msg = "training store is malformed"
            raise TrainingStoreError(msg) from error

    def save(self, state: TrainingState) -> None:
        """Atomically save bounded state unless ephemeral operation was requested."""
        if self.ephemeral:
            return
        state.validate()
        parent = self._validate_directory(create=True)
        if self.path.exists():
            self._validate_file()
        payload = json.dumps(
            _encode_state(state),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        descriptor = -1
        temporary = ""
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=".training-", dir=parent)
            os.fchmod(descriptor, _FILE_MODE)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            Path(temporary).replace(self.path)
            temporary = ""
            directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as error:
            msg = "could not atomically save training data"
            raise TrainingStoreError(msg) from error
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary:
                with contextlib.suppress(FileNotFoundError):
                    Path(temporary).unlink()

    def reset(self) -> None:
        """Remove persisted training state without touching ephemeral paths."""
        if self.ephemeral or not self.path.exists():
            return
        self._validate_directory(create=False)
        self._validate_file()
        try:
            self.path.unlink()
        except OSError as error:
            msg = "could not reset training data"
            raise TrainingStoreError(msg) from error

    def _validate_directory(self, *, create: bool) -> Path:
        parent = self.path.parent
        if create and not parent.exists():
            try:
                parent.mkdir(mode=_DIRECTORY_MODE, parents=True)
                parent.chmod(_DIRECTORY_MODE)
            except OSError as error:
                msg = "could not create the training directory"
                raise TrainingStoreError(msg) from error
        try:
            metadata = parent.lstat()
        except OSError as error:
            msg = "training directory is unavailable"
            raise TrainingStoreError(msg) from error
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            msg = "training directory must not be a symlink"
            raise TrainingStoreError(msg)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            msg = "training directory must be owner-only"
            raise TrainingStoreError(msg)
        return parent

    def _validate_file(self) -> None:
        metadata = self.path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            msg = "training store must be a regular file"
            raise TrainingStoreError(msg)
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            msg = "training store must be owner-only"
            raise TrainingStoreError(msg)


def _default_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return root / "gazeebo" / "training-v1.json"


def _encode_state(state: TrainingState) -> dict[str, object]:
    return {
        "version": STORE_VERSION,
        "next_sequence": state.next_sequence,
        "targets": [asdict(target) for target in state.targets],
        "clusters": [asdict(cluster) for cluster in state.clusters],
        "models": state.models,
        "validations": [asdict(validation) for validation in state.validations],
    }


def _decode_state(value: object) -> TrainingState:
    if not isinstance(value, dict):
        msg = "training store root must be an object"
        raise TrainingStoreError(msg)
    raw = cast("dict[str, object]", value)
    version = raw.get("version")
    if version == 0:
        raw = {
            "version": STORE_VERSION,
            "next_sequence": raw.get("next_sequence", 0),
            "targets": raw.get("targets", []),
            "clusters": [],
            "models": {},
            "validations": [],
        }
    elif version != STORE_VERSION:
        msg = "training store version is unsupported"
        raise TrainingStoreError(msg)

    state = TrainingState(
        next_sequence=_integer(raw.get("next_sequence", 0)),
        targets=[_decode_target(item) for item in _list(raw.get("targets", []))],
        clusters=[_decode_cluster(item) for item in _list(raw.get("clusters", []))],
        models=cast("dict[str, dict[str, object]]", raw.get("models", {})),
        validations=[_decode_validation(item) for item in _list(raw.get("validations", []))],
    )
    state.validate()
    return state


def _decode_target(value: object) -> StoredTarget:
    raw = _mapping(value)
    outputs = tuple(
        OutputDescriptor(
            key=str(output["key"]),
            x=_integer(output["x"]),
            y=_integer(output["y"]),
            width=_integer(output["width"]),
            height=_integer(output["height"]),
        )
        for output in (_mapping(item) for item in _list(raw["outputs"]))
    )
    return StoredTarget(
        sequence=_integer(raw["sequence"]),
        camera_id=str(raw["camera_id"]),
        feature_schema=str(raw["feature_schema"]),
        features=_float_tuple(raw["features"]),
        context=_float_tuple(raw["context"]),
        outputs=outputs,
        output_key=str(raw["output_key"]),
        target_u=_number(raw["target_u"]),
        target_v=_number(raw["target_v"]),
        desktop_u=_number(raw["desktop_u"]),
        desktop_v=_number(raw["desktop_v"]),
        zone=str(raw["zone"]),
    )


def _decode_cluster(value: object) -> ContextCluster:
    raw = _mapping(value)
    return ContextCluster(
        cluster_id=str(raw["cluster_id"]),
        camera_id=str(raw["camera_id"]),
        feature_schema=str(raw["feature_schema"]),
        centroid=_float_tuple(raw["centroid"]),
        variance=_float_tuple(raw["variance"]),
        sample_count=_integer(raw["sample_count"]),
        target_sequences=tuple(_integer(item) for item in _list(raw["target_sequences"])),
        median_error=_optional_float(raw.get("median_error")),
        edge_error=_optional_float(raw.get("edge_error")),
    )


def _decode_validation(value: object) -> ValidationSummary:
    raw = _mapping(value)
    return ValidationSummary(
        sequence=_integer(raw["sequence"]),
        camera_id=str(raw["camera_id"]),
        topology_id=str(raw["topology_id"]),
        routing=str(raw["routing"]),
        median_error=_number(raw["median_error"]),
        edge_error=_number(raw["edge_error"]),
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = "training record must be an object"
        raise TrainingStoreError(msg)
    return cast("dict[str, object]", value)


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        msg = "training record must contain arrays"
        raise TrainingStoreError(msg)
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        msg = "training record integer is malformed"
        raise TrainingStoreError(msg)
    return int(value)


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        msg = "training record number is malformed"
        raise TrainingStoreError(msg)
    return float(value)


def _float_tuple(value: object) -> tuple[float, ...]:
    return tuple(_number(item) for item in _list(value))


def _optional_float(value: object) -> float | None:
    return None if value is None else _number(value)
