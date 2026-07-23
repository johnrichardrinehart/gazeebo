"""Tests for secure target-level training persistence."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from gazeebo.state import (
    MAXIMUM_STORED_TARGETS,
    ContextCluster,
    OutputDescriptor,
    StoredTarget,
    TrainingState,
    TrainingStore,
    TrainingStoreError,
    ValidationSummary,
)


def target(sequence: int = 0) -> StoredTarget:
    """Create one finite target-level aggregate."""
    return StoredTarget(
        sequence=sequence,
        camera_id="camera-a",
        feature_schema="gaze-v1",
        features=(0.1, 0.2, 0.3),
        context=(0.0, 0.1, 0.5, 0.4),
        outputs=(OutputDescriptor("left", 0, 0, 1000, 700),),
        output_key="left",
        target_u=0.5,
        target_v=0.5,
        desktop_u=0.5,
        desktop_v=0.5,
        zone="center",
    )


def state() -> TrainingState:
    """Create a state containing every serialized record type."""
    return TrainingState(
        next_sequence=1,
        targets=[target()],
        clusters=[
            ContextCluster(
                "context-0",
                "camera-a",
                "gaze-v1",
                (0.0, 0.1, 0.5, 0.4),
                (0.01, 0.01, 0.01, 0.01),
                1,
                (0,),
                median_error=80.0,
                edge_error=90.0,
            )
        ],
        models={"global": {"kind": "affine", "coefficients": [[1.0, 2.0]]}},
        validations=[ValidationSummary(0, "camera-a", "layout-a", "global", 80.0, 90.0)],
    )


class TrainingStoreTests(unittest.TestCase):
    """Lock schema, permissions, atomicity, migration, and ephemeral behavior."""

    def test_round_trip_uses_owner_only_file_and_directory(self) -> None:
        """Persisted derived data are private and survive a complete reload."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gazeebo" / "training-v1.json"
            store = TrainingStore(path)
            expected = state()
            store.save(expected)

            assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
            assert store.load() == expected
            assert list(path.parent.glob(".training-*")) == []

    def test_unsupported_corrupt_and_insecure_state_fails_closed(self) -> None:
        """Malformed or publicly readable state is never partly accepted."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            path.write_text('{"version":99}', encoding="utf-8")
            path.chmod(0o600)
            with self.assertRaisesRegex(TrainingStoreError, "version"):
                TrainingStore(path).load()

            path.write_text("not json", encoding="utf-8")
            with self.assertRaisesRegex(TrainingStoreError, "malformed"):
                TrainingStore(path).load()

            path.write_text('{"version":1}', encoding="utf-8")
            path.chmod(0o644)
            with self.assertRaisesRegex(TrainingStoreError, "owner-only"):
                TrainingStore(path).load()

    def test_symlinked_store_is_rejected(self) -> None:
        """A store cannot redirect sensitive writes through a symbolic link."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "elsewhere"
            destination.write_text('{"version":1}', encoding="utf-8")
            destination.chmod(0o600)
            path = root / "training-v1.json"
            path.symlink_to(destination)
            with self.assertRaisesRegex(TrainingStoreError, "regular file"):
                TrainingStore(path).load()

    def test_version_zero_empty_state_migrates_without_writing(self) -> None:
        """The only pre-release schema migrates deterministically in memory."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            original = json.dumps({"version": 0, "next_sequence": 7, "targets": []})
            path.write_text(original, encoding="utf-8")
            path.chmod(0o600)
            migrated = TrainingStore(path).load()
            assert migrated.next_sequence == 7
            assert migrated.targets == []
            assert path.read_text(encoding="utf-8") == original

    def test_ephemeral_mode_does_not_read_write_or_reset_path(self) -> None:
        """Ephemeral operation ignores even malformed persistent state."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            path.write_text("private existing bytes", encoding="utf-8")
            path.chmod(0o644)
            store = TrainingStore(path, ephemeral=True)
            assert store.load() == TrainingState()
            store.save(state())
            store.reset()
            assert path.read_text(encoding="utf-8") == "private existing bytes"

    def test_reset_removes_only_a_valid_store(self) -> None:
        """The reset operation removes private state and remains idempotent."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training-v1.json"
            store = TrainingStore(path)
            store.save(state())
            store.reset()
            store.reset()
            assert not path.exists()

    def test_store_rejects_unbounded_target_data(self) -> None:
        """Persistence cannot grow beyond its documented target limit."""
        oversized = TrainingState(
            next_sequence=MAXIMUM_STORED_TARGETS + 1,
            targets=[target(index) for index in range(MAXIMUM_STORED_TARGETS + 1)],
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(TrainingStoreError, "target limit"),
        ):
            TrainingStore(Path(temporary) / "training-v1.json").save(oversized)

    def test_directory_permissions_are_checked_before_loading(self) -> None:
        """A shared training directory cannot expose target-level features."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "gazeebo"
            directory.mkdir(mode=0o700)
            path = directory / "training-v1.json"
            path.write_text('{"version":1}', encoding="utf-8")
            path.chmod(0o600)
            directory.chmod(0o755)
            with self.assertRaisesRegex(TrainingStoreError, "owner-only"):
                TrainingStore(path).load()
            directory.chmod(0o700)


if __name__ == "__main__":
    unittest.main()
