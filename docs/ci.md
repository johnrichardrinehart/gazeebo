# CI boundary

CI configuration is an adapter, not the project build system.

## Adapter responsibilities

A CI platform file may:

- check out the requested revision;
- install a pinned Nix implementation and version;
- enable required Nix features, sandboxing, and virtualization;
- authenticate input fetches and binary caches;
- connect trusted remote builders;
- restore and publish Nix store paths;
- pass explicitly authorized secrets to one command;
- retain logs and declared build artifacts.

These operations necessarily depend on the runner and provider. Keep them declarative and use maintained provisioning/cache actions or images where available.

## Repository responsibilities

The repository owns:

- build and test selection;
- formatter and linter configuration;
- concurrency that is intrinsic to the build;
- retries required by project semantics;
- artifact construction and validation;
- deployment behavior;
- cleanup and rollback behavior.

Prefer a CI execution step containing exactly:

```yaml
- run: nix flake check --print-build-logs
```

Do not hide this standard command behind a repository-specific script, package, or generic `ci` app. The extra name obscures what CI does without adding reproducibility.

When an additional operation is impure, credentialed, or unsuitable for the flake-check sandbox, package that operation under a descriptive name and invoke it explicitly, for example:

```yaml
- run: nix run .#integration-test
```

A provider matrix may invoke multiple descriptively named flake apps, but each invocation should remain independently reproducible outside the CI service.

## Packaged shell entry points

Shell is appropriate for substantive process orchestration and external command composition. Package such behavior with `writeShellApplication` so that:

- runtime executables are explicit store dependencies;
- ShellCheck runs while building the package;
- strict mode is supplied consistently;
- CI does not need to install an ad hoc tool list;
- `nix run .#name` works locally and on every provider.

If several operations share substantial project-specific behavior, package a shared script or executable rather than sourcing files installed by CI setup steps. A one-command forwarder is not substantial shared behavior and should be deleted in favor of the command it invokes.

## Nix provisioning

Pin the provisioning adapter to obtain a predictable Nix version. The adapter should configure:

- `nix-command` and `flakes`;
- sandboxing appropriate to the runner;
- KVM when NixOS VM tests require it;
- authenticated Git access when private inputs are intentional;
- trusted substituters and public keys;
- builders and `builders-use-substitutes` where remote builders are used.

Project logic must not assume a mutable runner image happens to contain the desired Nix version.

## Store-path caching

Cache Nix store paths rather than language-specific build directories whenever the build is represented by Nix derivations. A binary cache preserves content-addressed results across branches, jobs, machines, and CI runs without restoring mutable work directories.

A cache adapter must define:

- read availability for pull requests and forks;
- write authority for trusted branches;
- retention and size limits;
- cache poisoning protections;
- signing or trust configuration;
- whether results are available only to CI or also to developers and users.

GitHub's native Actions cache is suitable for CI-only reuse. Cachix, FlakeHub Cache, Attic, and ordinary Nix binary caches can also serve developers or deployment hosts. Remote builders complement caches by moving computation; they do not replace cache publication.

### Measure the cache adapter

The template deliberately provides `ci-magic-cache.yml` and `ci-no-cache.yml`. Both provision the same Nix release and execute the same flake command. Compare at least:

- total job duration;
- cache setup and post-job publication duration;
- the project-command duration;
- cold and warm revisions;
- bytes and entries published;
- cache API throttling or eviction evidence.

Magic Nix Cache v14 serializes complete store-path uploads through one worker. Its fixed concurrency of four applies to chunks within one file, not to multiple paths, and is not configurable. Because each path generally creates a NAR and `.narinfo` cache entry, broad closures can approach GitHub's 200-entry-per-minute repository limit and make publication a net loss. Select it only when observed reuse outweighs that cost.

## Credentials

Provisioning credentials should be narrowly scoped to the adapter that consumes them. Application and deployment secrets should be declared through SecretSpec and exposed only to the packaged command that needs them.

Untrusted pull requests must not receive write-capable cache credentials or deployment secrets. Their derivations may consume trusted cache entries but must not be allowed to publish entries trusted by protected branches without an isolation policy.
