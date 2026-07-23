"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING

from gazeebo import __version__
from gazeebo.camera import CameraError, OpenCVCamera
from gazeebo.contexts import build_router
from gazeebo.contracts import RuntimeStatus
from gazeebo.control import ControlError, TrainingControl, request_training
from gazeebo.game import GameConfig, GameError, LayerShellCalibrationGame
from gazeebo.geometry import DisplayTopology
from gazeebo.hud import LayerShellDebugHud
from gazeebo.portal import PortalError, PortalPointerController
from gazeebo.runtime import (
    FEATURE_SCHEMA,
    ConsoleStatus,
    TrackingConfig,
    TrackingError,
    install_signal_handlers,
    run_owned_session,
)
from gazeebo.state import TrainingState, TrainingStore, TrainingStoreError
from gazeebo.vision import OpenSeeFaceEstimator, VisionError

if TYPE_CHECKING:
    from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="gazeebo",
        description="Local gaze-driven cursor navigation",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "train", "reset-training"),
        default="run",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--camera", help="V4L2 device path or numeric index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--vision-confidence", type=float, default=0.55)
    parser.add_argument("--open-eye-threshold", type=float, default=0.35)
    parser.add_argument("--calibration-settle", type=float, default=1.00)
    parser.add_argument("--calibration-samples", type=int, default=8)
    parser.add_argument("--startup-context-samples", type=int, default=8)
    parser.add_argument("--smoothing-alpha", type=float, default=0.35)
    parser.add_argument("--smoothing-dead-zone", type=float, default=6.0)
    parser.add_argument("--smoothing-maximum-step", type=float, default=600.0)
    parser.add_argument(
        "--pointer-update-interval",
        type=float,
        default=0.10,
        help="minimum seconds between pointer moves; zero updates continuously",
    )
    parser.add_argument("--game-batch-size", type=int, default=5)
    parser.add_argument("--game-precision-threshold", type=float, default=100.0)
    parser.add_argument("--game-maximum-targets", type=int, default=55)
    parser.add_argument("--game-settle", type=float, default=0.75)
    parser.add_argument("--game-dwell", type=float, default=1.25)
    parser.add_argument("--game-target-timeout", type=float, default=6.0)
    parser.add_argument("--game-minimum-diameter", type=float, default=48.0)
    parser.add_argument("--game-maximum-diameter", type=float, default=144.0)
    parser.add_argument(
        "--ephemeral",
        action="store_true",
        help="do not read or write local target-level training data",
    )
    parser.add_argument(
        "--debug-hud",
        action="store_true",
        help="show authorized regions, routing, and cursor coordinates once per second",
    )
    return parser


def _camera_device(value: str | None) -> str | int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _tracking_config(arguments: argparse.Namespace) -> TrackingConfig:
    return TrackingConfig(
        calibration_settle_seconds=arguments.calibration_settle,
        calibration_samples_per_target=arguments.calibration_samples,
        startup_context_samples=arguments.startup_context_samples,
        smoothing_alpha=arguments.smoothing_alpha,
        smoothing_dead_zone=arguments.smoothing_dead_zone,
        smoothing_maximum_step=arguments.smoothing_maximum_step,
        pointer_update_interval_seconds=arguments.pointer_update_interval,
        open_eye_threshold=arguments.open_eye_threshold,
    )


def _game_config(arguments: argparse.Namespace) -> GameConfig:
    return GameConfig(
        batch_size=arguments.game_batch_size,
        precision_threshold=arguments.game_precision_threshold,
        maximum_targets=arguments.game_maximum_targets,
        settle_seconds=arguments.game_settle,
        dwell_seconds=arguments.game_dwell,
        target_timeout_seconds=arguments.game_target_timeout,
        minimum_diameter=arguments.game_minimum_diameter,
        maximum_diameter=arguments.game_maximum_diameter,
    )


def _open_vision(arguments: argparse.Namespace) -> tuple[OpenCVCamera, OpenSeeFaceEstimator]:
    camera = OpenCVCamera(
        _camera_device(arguments.camera),
        width=arguments.width,
        height=arguments.height,
        frames_per_second=arguments.fps,
    )
    try:
        vision = OpenSeeFaceEstimator(
            *camera.dimensions,
            minimum_confidence=arguments.vision_confidence,
        )
    except Exception:
        camera.close()
        raise
    return camera, vision


async def _open_startup_resources(  # noqa: C901, PLR0912
    arguments: argparse.Namespace,
    stop: asyncio.Event,
) -> tuple[PortalPointerController, OpenCVCamera, OpenSeeFaceEstimator] | None:
    """Authorize geometry while camera and vision initialize, unless stopped."""
    portal_task = asyncio.create_task(PortalPointerController.authorize())
    vision_task = asyncio.create_task(asyncio.to_thread(_open_vision, arguments))
    stop_task = asyncio.create_task(stop.wait())
    pending: set[asyncio.Task[object]] = {portal_task, vision_task, stop_task}
    while not portal_task.done() or not vision_task.done():
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done:
            if not portal_task.done():
                portal_task.cancel()
            break
    if not vision_task.done():
        await asyncio.wait((vision_task,))
    if not portal_task.done():
        portal_task.cancel()
    portal_result, vision_result = await asyncio.gather(
        portal_task,
        vision_task,
        return_exceptions=True,
    )
    stop_task.cancel()
    await asyncio.gather(stop_task, return_exceptions=True)
    pointer = portal_result if isinstance(portal_result, PortalPointerController) else None
    resources = vision_result if isinstance(vision_result, tuple) else None
    if stop.is_set():
        if resources is not None:
            resources[1].close()
            resources[0].close()
        if pointer is not None:
            await pointer.close()
        return None
    if isinstance(portal_result, BaseException) or isinstance(vision_result, BaseException):
        if resources is not None:
            resources[1].close()
            resources[0].close()
        if pointer is not None:
            await pointer.close()
        error = portal_result if isinstance(portal_result, BaseException) else vision_result
        if not isinstance(error, BaseException):
            msg = "parallel startup failed without an exception"
            raise TypeError(msg)
        raise error
    if pointer is None or resources is None:
        msg = "parallel startup returned incomplete resources"
        raise RuntimeError(msg)
    return pointer, resources[0], resources[1]


def _needs_training(
    state: TrainingState,
    pointer: PortalPointerController,
    camera: OpenCVCamera,
) -> bool:
    try:
        build_router(
            state,
            DisplayTopology(pointer.regions),
            camera_id=camera.camera_id,
            feature_schema=FEATURE_SCHEMA,
        )
    except ValueError:
        return True
    return False


async def _run(  # noqa: C901, PLR0911, PLR0912, PLR0915
    arguments: argparse.Namespace,
) -> int:
    status = ConsoleStatus()
    status.report(RuntimeStatus.STARTING)
    tracking = _tracking_config(arguments)
    game_config = _game_config(arguments)
    store = TrainingStore(ephemeral=arguments.ephemeral)
    stop = asyncio.Event()
    install_signal_handlers(stop)
    training_requested = asyncio.Event()
    control = TrainingControl(training_requested)
    pointer: PortalPointerController | None = None
    hud: LayerShellDebugHud | None = None
    game: LayerShellCalibrationGame | None = None
    camera: OpenCVCamera | None = None
    vision: OpenSeeFaceEstimator | None = None
    session_started = False
    try:
        status.report(RuntimeStatus.LOADING)
        if arguments.command == "reset-training":
            store.reset()
            status.report(RuntimeStatus.STOPPED)
            return 0
        if arguments.command == "train" and await request_training():
            status.report(RuntimeStatus.STOPPED, "active session accepted training request")
            return 0
        state = store.load()
        if stop.is_set():
            return 0
        await control.start()
        status.report(RuntimeStatus.AUTHORIZING)
        resources = await _open_startup_resources(arguments, stop)
        if resources is None:
            return 0
        pointer, camera, vision = resources
        if arguments.debug_hud:
            hud = LayerShellDebugHud.create(pointer.regions)
        train_requested = arguments.command == "train"
        if train_requested or _needs_training(state, pointer, camera):
            game = LayerShellCalibrationGame.create(pointer.regions)
        session_started = True
        return await run_owned_session(
            camera,
            vision,
            pointer,
            status,
            stop,
            hud=hud,
            game=game,
            tracking=tracking,
            game_config=game_config,
            training_store=store,
            training_state=state,
            train_requested=train_requested,
            training_requested_event=training_requested,
            game_factory=LayerShellCalibrationGame.create,
            training_control=control,
        )
    except ControlError as error:
        status.report(RuntimeStatus.INPUT_ERROR, str(error))
        return 2
    except PortalError as error:
        status.report(RuntimeStatus.INPUT_ERROR, str(error))
        return 2
    except TrainingStoreError as error:
        status.report(RuntimeStatus.STORE_ERROR, str(error))
        return 1
    except GameError as error:
        status.report(RuntimeStatus.GAME_ERROR, str(error))
        return 1
    except (CameraError, VisionError, TrackingError) as error:
        status.report(RuntimeStatus.CAMERA_ERROR, str(error))
        return 1
    finally:
        if not session_started:
            await control.close()
            if vision is not None:
                vision.close()
            if camera is not None:
                camera.close()
            if game is not None:
                await game.close()
            if hud is not None:
                await hud.close()
            if pointer is not None:
                await pointer.close()
            if arguments.command != "reset-training":
                status.report(RuntimeStatus.STOPPED)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one foreground navigation or training command."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return asyncio.run(_run(arguments))
    except ValueError as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        return 130
