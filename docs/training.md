# Automatic training and model selection

Gazeebo keeps one local training store. Users do not create, name, or select profiles. The store learns several operating contexts and chooses among them from camera observations and the portal-authorized display layout.

## Stored data

The versioned store lives below `$XDG_DATA_HOME/gazeebo`, or `$HOME/.local/share/gazeebo` when `XDG_DATA_HOME` is unset. Its directory is mode `0700` and its files are mode `0600`. Updates write a complete candidate to a sibling temporary file, synchronize it, replace the prior file atomically, and synchronize the directory. A malformed, unsupported, symlinked, or insecure store fails closed instead of being partly loaded.

Each accepted target contributes one record:

- the feature-schema version and an opaque camera fingerprint;
- one median gaze feature vector;
- a context vector containing head pose, normalized face center and size, and frame luminance mean and spread;
- a generic output descriptor;
- the target's normalized position within that output and its center, edge, or corner zone;
- a fallback target position normalized to the source topology's bounding rectangle;
- a monotonic sequence number used for deterministic retention.

The store also contains bounded context-cluster statistics, serialized global and local estimator parameters, topology descriptors, and aggregate holdout results. It never contains frames, images, raw landmarks, frame-level feature streams, personal names, device paths, or wall-clock usage history.

`gazeebo reset-training` removes the store. Ephemeral operation neither reads nor writes it. Normal startup creates no new training records; only a completed initial or on-demand training flow can update the store.

## Display topology adaptation

The portal remains the authority for pointer access and logical geometry. A topology descriptor records each authorized region's opaque mapping identifier when available, logical size and position, and relative ordering. Target labels remain output-relative so resolution and position changes do not make every prior sample stale.

Gazeebo matches stored and current outputs in this order:

1. An unambiguous portal mapping identifier.
2. A unique logical-size and relative-neighborhood match.
3. No output match.

A strong match maps the stored normalized output target into the current region. Removed outputs contribute no samples. An added output has no local samples, but existing samples still fit the global and compatible context models over the current logical geometry. Samples without an unambiguous output match use their topology-bounds-normalized fallback label and are marked weak. Predictions always pass through authorized-union projection.

Exact and strong matches may report prior measured quality with its source topology. Weak matches report inferred, unvalidated quality. Gazeebo does not claim that logical geometry reveals physical display size, viewing distance, or a monitor's real-world placement. The fallback provides best-effort navigation until the user invokes training.

Portal authorization and geometry are mandatory before pointer motion. A closed portal session stops motion even when a stored model is available.

## Context vectors and clusters

Routing context excludes pupil coordinates so looking in a new direction does not itself select a different model. The context uses:

- head pitch, yaw, and roll;
- normalized face center, width, and height;
- frame luminance mean and standard deviation;
- the feature schema and opaque camera fingerprint as compatibility boundaries.

An online cluster stores a centroid, diagonal variance, sample count, validation quality, and a bounded balanced coreset. Distances use fixed dimension weights and variance floors. Assignment is deterministic: use the nearest compatible cluster within the configured distance, otherwise create a cluster. At the cluster limit, merge the closest compatible pair when their fitted predictions agree; otherwise evict the least useful redundant cluster into the global coreset. No camera/schema partition can exceed eight clusters.

Each cluster keeps at most 64 targets, and the global coreset keeps at most 256. Retention buckets samples by output descriptor and center, edge, or corner zone. Within a bucket, deterministic farthest-point selection retains context and gaze-feature diversity. This prevents one recent posture, one display, or repeated center targets from replacing all other evidence.

## Global and local estimators

The global estimator uses the balanced coreset across every compatible context. A local expert uses one context cluster. Both retain the existing held-out affine/RBF feature-set selection.

At startup Gazeebo gathers a short passive context window while portal geometry and model data finish loading. It scores compatible experts by context distance, topology strength, and prior holdout quality. The router blends the best two experts with the global estimator. Exponentially smoothed weights and a switching margin prevent frame-to-frame model changes from jumping the cursor. An out-of-distribution context increases the global weight and reports `training-recommended` rather than inventing measured accuracy.

The global estimator is always the fallback. Local experts model posture and illumination residuals without making users understand clusters.

## Training transactions

A first run without usable training data enters initial training. `gazeebo train` requests training from an active foreground process through an owner-only Unix socket; if no process is active, it starts a foreground training session. There is no daemon.

Each target is reduced to one median gaze and context record. A batch is scored before any record from that batch trains a candidate. A failed, already reported batch may enter the candidate coreset. The next batch remains unseen. The final successful or terminal batch never becomes training data.

Before committing a candidate store, Gazeebo evaluates both incumbent and candidate routing on the same unseen batch. A candidate may replace an incumbent only when it does not worsen either median radial error or edge/corner error. A new camera, context, or topology expert with no compatible incumbent may be established from a finite unseen result; when a global fallback exists, the candidate must also pass the same no-regression comparison. A terminal precision miss may establish the first finite model so later invocations can improve it, but it remains measured as below threshold and Gazeebo recommends more training. Rejected candidates leave the prior store byte-for-byte unchanged.

The default batch size is five, the precision threshold is 100 logical pixels, and one invocation stops after at most 55 targets. Repeated invocations can add evidence while each transaction remains finite.

## Startup and lifecycle

Store loading, vision-model loading, portal authorization, and camera initialization run concurrently where ownership and toolkit constraints permit. Camera observations may supply passive context during the foreground invocation. No pointer message is sent until the portal has authorized pointer access, returned valid regions, and model routing has completed.

The active process owns its camera, portal session, native surfaces, model objects, and runtime socket. Normal exit, errors, `SIGINT`, and `SIGTERM` close all of them through one idempotent path. The socket and training surfaces do not outlive the process.
