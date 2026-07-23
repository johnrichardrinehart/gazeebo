# Gazeebo

Gazeebo is a local, process-scoped gaze-navigation tool for Linux desktops. It uses a webcam to move the cursor across every display authorized through the desktop portal. It emits cursor motion only: no eye gesture, pointer-button, keyboard, scrolling, drag, or dwell-click action exists.

## Runtime requirements

Gazeebo needs:

- Linux with a desktop portal that supports RemoteDesktop pointer authorization and returns logical geometry for every selected monitor;
- a Wayland session with layer shell for initial and on-demand target training;
- a locally attached camera supported by OpenCV and V4L2.

The portal selector defines the authorized displays. Gazeebo does not read compositor configuration or invoke a window-manager command. A portal backend that omits multi-display positions fails safely.

## Run

```console
gazeebo
gazeebo --camera /dev/video2
gazeebo --camera 2
```

Startup concurrently loads local training state and vision resources while requesting portal authorization where safe. Cursor motion begins only after pointer access and valid authorized geometry are available.

On the first run, Gazeebo shows red targets across the authorized displays, trains a local model, validates it on unseen targets, and saves accepted target-level data. Later runs passively infer posture and illumination, select or blend automatic context experts, and begin navigation without mandatory target validation.

Predictions can cross authorized displays. Predictions in desktop gaps or outside the selected union project to the nearest authorized display. Motion defaults to at most one update every 100 milliseconds; `--pointer-update-interval 0` requests continuous updates.

## Add training data

```console
gazeebo train
```

If Gazeebo is active, this command asks that foreground process to enter training through an owner-only Unix socket. Otherwise it starts a foreground training session. No daemon or helper remains afterward.

Training shows one click-through, keyboard-inactive red target at a time. Stable, confident, open-eye dwell samples advance automatically. Five-target batches report hit rate, median radial error, edge/corner error, response time, and model routing. A reported batch can train a candidate; a fresh unseen batch decides whether the candidate is safe to save. The final holdout never trains the result it reports.

The default precision threshold is 100 logical pixels and one invocation stops after at most 55 targets. Batch size, precision threshold, maximum targets, settle and dwell timing, timeout, and diameter range are configurable.

## Automatic training store

Users do not create or select profiles. Gazeebo keeps one automatic store below `$XDG_DATA_HOME/gazeebo`, or `$HOME/.local/share/gazeebo` when `XDG_DATA_HOME` is unset.

The store contains bounded target-level median gaze features, normalized target labels, posture and illumination aggregates, automatic context-cluster statistics, fitted coefficients, generic camera and topology descriptors, and aggregate holdout measurements. It never contains frames, images, video, or raw frame-level landmarks. The directory and files are owner-only, and updates replace the complete state atomically.

Stored targets remain output-relative. Gazeebo remaps them after unambiguous monitor resolution, scale, or position changes; excludes removed-output samples; and uses global/context fallback models for added or ambiguous outputs. Weak topology matches are reported as inferred and unvalidated. Every result still passes through authorized-union clipping.

Reset all local training data:

```console
gazeebo reset-training
```

Run without reading or writing training state:

```console
gazeebo --ephemeral
```

## Debug HUD

```console
gazeebo --debug-hud
```

The opt-in HUD lists every authorized region, the active region, global cursor coordinates, selected model blend, and topology quality. It is transparent, read-only, always on top, click-through, and limited to one update per second. It never displays stored feature values.

## Privacy and lifecycle

Gazeebo has no telemetry, cloud client, or recording path. Frames and frame-level derived values stay in memory and are discarded after processing. Only documented, bounded target-level aggregates enter the local training store.

One foreground process owns the camera, portal session, models, native surfaces, and runtime socket. Normal exit, errors, `SIGINT`, and `SIGTERM` release them through the same cleanup path.

## Development

The root flake is consumer-clean and exports the package, app, overlay, and NixOS module. Development tooling lives in the `dev` partition.

```console
nix develop
nix fmt
nix flake check --print-build-logs
```

See [`docs/architecture.md`](docs/architecture.md) for runtime boundaries and [`docs/training.md`](docs/training.md) for storage, topology adaptation, clustering, routing, retention, and candidate acceptance.
