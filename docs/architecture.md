# Architecture

## Runtime boundary

`gazeebo` is one foreground process. Startup owns one camera, one desktop-portal session, the selected gaze models, an optional training surface and debug HUD, and an owner-only runtime control socket. Normal completion, `SIGINT`, `SIGTERM`, a portal-session closure, or an unrecoverable camera or store error runs the same idempotent cleanup path. No service, daemon, renderer, or worker process survives the executable.

The application separates hardware and desktop APIs from deterministic policy:

- `CameraCapture` owns a local camera and yields in-memory frames.
- `VisionEstimator` turns a frame into open-eye confidence, gaze features, and a pupil-independent posture and illumination context.
- `PointerController` exposes every portal-authorized display and region-local absolute motion. It has no button API.
- Pure calibration, topology adaptation, context clustering, model routing, clipping, and smoothing components consume those interfaces.
- `TrainingStore` owns bounded target-level derived data and atomic local updates.
- `StatusSink` reports state without coupling policy to a shell or graphical toolkit.
- `DebugHud` publishes opt-in, rate-limited diagnostics without changing navigation policy.

Tests replace every interface with an in-memory implementation. Camera devices, a graphical session, and user authorization are required only for explicit hardware acceptance.

## Camera and vision

OpenCV's [`VideoCapture`](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) API opens V4L2 capture devices and returns frames held in process memory. An explicit command-line device takes precedence. Automatic selection probes local capture devices and rejects sources that cannot return a frame; it does not encode a device path or product name.

[OpenSeeFace](https://github.com/emilianavt/OpenSeeFace) supplies the CPU face detector, 66-point landmarks, head pose, and pupil estimates. The package uses the pinned Nixpkgs derivation rather than downloading models at runtime, and imports the tracker in process instead of starting its UDP server.

The adapter selects one confident face. It converts normalized pupil coordinates into gaze features and derives routing context from head pose, normalized face position and size, and aggregate frame luminance. Context excludes pupil position so normal gaze changes cannot switch experts. Frames and frame-level landmarks are discarded after each observation.

## Authorized cursor motion

The [XDG RemoteDesktop portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.RemoteDesktop.html) is the pointer backend. Gazeebo requests pointer control, not keyboard or touchscreen control, and emits only `NotifyPointerMotionAbsolute`. It has no gesture recognizer, pointer-button API, click action, or evdev button mapping.

The request includes monitor sources from the [ScreenCast portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.ScreenCast.html), because absolute motion addresses an authorized stream. The portal returns one stream and property map per authorized display. `position` and `size` describe logical geometry; `logical_size` takes precedence when supplied. Multi-display sessions require stream positions so Gazeebo can reconstruct the authorized union. Opaque portal mapping identifiers become generic region descriptors when available.

Predictions inside an authorized region retain their global logical coordinate. Predictions outside the union, including desktop gaps, project to the nearest region edge and then convert to stream-local coordinates. Exponential smoothing, a dead zone, a maximum step, and a default 100-millisecond pointer interval reduce jitter and sudden movement. An interval of zero requests continuous updates.

Denied pointer access, incomplete geometry, malformed streams, or a closed portal session stops motion safely. Portal authorization and valid geometry are mandatory before the first pointer event. Gazeebo never opens `/dev/uinput` or calls a compositor-specific output API.

## Automatic training state

Gazeebo keeps one automatic local training store rather than exposing profiles. It persists bounded target-level median gaze features, pupil-independent context, output-relative labels, topology descriptors, model parameters, and aggregate validation results. It does not persist frames, images, raw landmarks, or frame-level observations.

The store supports a global fallback model and bounded context experts. Online clustering groups compatible camera observations by posture, face geometry, and illumination. A balanced coreset preserves display and center, edge, and corner coverage. Passive startup observations select or blend compatible experts; smoothed routing and hysteresis prevent model-switch jumps.

Targets are stored relative to an output and can be remapped after unambiguous logical resolution, scale, or position changes. Removed outputs are excluded. Added or ambiguous outputs use the global/context fallback and topology-bounds-normalized labels for best-effort navigation. Weak matches report inferred rather than measured quality. Every prediction still passes through authorized-union projection.

[`training.md`](training.md) specifies the schema, data minimization, topology matching, clustering, retention, model routing, and candidate acceptance rules.

## Initial and on-demand training

A first run without usable stored evidence enters initial training. Users can request more evidence with `gazeebo train`; they never name a profile. An active foreground process accepts the request through an owner-only Unix socket and transitions into training. If no process is active, the command owns a new foreground session. The socket is not a daemon and disappears on exit.

The native renderer creates one transparent overlay-layer surface per authorized output. Each surface spans only its output, has an empty input region, and requests no keyboard interaction. One surface draws the current red target while the others remain transparent. A deterministic sequence distributes varied circle diameters and center, edge, and corner positions across outputs.

For each target, the game waits for the settle interval and accepts a rolling dwell window only when observations are confident, open-eye, and stable. Component-wise medians reduce the accepted window to one target-level gaze and context record.

The game reports each five-target batch before using it for training. Reports include hit rate, median radial error, edge/corner error, response time, topology quality, and expert routing. A failed reported batch may train a candidate. A new unseen batch evaluates that candidate. The final successful or terminal batch remains a holdout.

Before a persistent update, incumbent and candidate routing are scored on the same unseen measurements. The candidate must not worsen either median or edge/corner error. A finite terminal result may establish a context that has no compatible model, remains labeled below threshold, and triggers a training recommendation; a rejected candidate leaves the prior store unchanged. The default precision threshold is 100 logical pixels and the default hard limit is 55 targets per invocation, including first-run anchors. Timing, batch size, diameter range, threshold, and limit are finite validated settings.

## Development HUD

`--debug-hud` loads the packaged native renderer into the process and creates a transparent generic layer-shell surface. An empty input region and disabled keyboard interactivity make it read-only and click-through. The renderer uses XDG Output to match global coordinates to generic output descriptions where available.

The HUD lists every authorized region, the current global logical X/Y coordinate, topology match quality, and selected global/expert blend. It never displays stored feature values. A one-second limiter prevents high-rate redraws. The HUD is disabled by default and destroyed through the shared cleanup path.

## Failure and privacy rules

The process reports loading, authorizing, selecting-model, topology-unvalidated, initial-training, adaptive-training, validating, active, training-recommended, stopped, and concrete camera, portal, model, store, or game failures. Missing or stale frames, unavailable confident face observations, invalid geometry, closed authorization, and unsafe store state stop or suspend motion rather than producing guesses outside the authorized union.

All processing remains local. The application has no telemetry, cloud client, video recording, or image persistence. The XDG training store contains only bounded target-level derived records and protected model state. Its directory and files are owner-only; updates are atomic. `gazeebo reset-training` removes it, while ephemeral mode neither reads nor writes it.

## Reproducible packaging

The root flake exports the Gazeebo package and app for `x86_64-linux`. Its explicit fileset contains only runtime source, metadata, documentation, and tests. Nixpkgs supplies Python, dbus-next, NumPy, OpenCV, ONNX Runtime, OpenSeeFace, Wayland, Cairo, and Pango. Development formatters, hooks, type checking, and CI inputs remain in the `dev` flake partition.

Authoritative checks run deterministic tests, package/import checks, formatting and linting, strict type checking, and source-policy checks. Portal, camera, physical topology, posture, illumination, and target accuracy remain explicit hardware acceptance work because a sandbox cannot provide those observations.
