# Customizing the template

## Outputs

Add project deliverables as they acquire real behavior. Do not wrap standard commands such as `nix flake check`, `nix fmt`, or `nix develop` in project-specific package or app names. When work cannot be represented as a flake check, expose it through a narrowly named app such as `integration-test` or `deploy` rather than a generic `ci` entry point.

Move substantial output wiring into the matching directory under `nix/`:

- `packages/` for build products and packaged commands;
- `checks/` for deterministic release gates;
- `apps/` for nontrivial app definitions;
- `overlays/` and system-module outputs only when exported to consumers;
- `lib/` only for logic reused by multiple expressions.

Keep consumer inputs in the root flake. Keep formatters, hooks, language servers, and authoring utilities in the development subflake.

## Systems

The template explicitly lists Linux systems. Change the list to the platforms actually supported and checked. A shared `nix-systems` input can be appropriate when several flakes intentionally follow one system policy; a literal list better communicates deliberately narrow support.

## Source filesets

Do not use the whole checkout as package source by default. Select the source directory, manifests, lock files, generated inputs, and build metadata required by each derivation. A package with different inputs should have a different fileset.

## Format inventory

After adding project files, list tracked extensions and special names, then update treefmt:

```console
git ls-files | awk -F/ '{ print $NF }' | sed -n 's/.*\.//p' | sort -u
```

Classify extensionless scripts, `.envrc`, generated files, and vendored trees separately. Do not allow two mutating formatters to target the same files unless their order is intentional and stable.

Configure file-oriented formatters and linters through treefmt-nix, including tools such as deadnix, statix, and ShellCheck. The Git hook configuration should enable only the treefmt hook for that policy; do not enable the same tools again as individual git-hooks.nix hooks. Direct hooks are reserved for checks outside treefmt's responsibility.

This template sets `treefmt.flakeCheck = false` while retaining the git-hooks.nix check. The latter runs the treefmt hook in a pure, sandboxed derivation together with Git-specific hooks, so a separate `checks.treefmt` would duplicate work. Do not disable `pre-commit.check` without either restoring `treefmt.flakeCheck` or defining an equivalent sandboxed treefmt check.

## Checks

Add the cheapest precise check for each behavior:

- pure Nix unit tests for Nix functions;
- package tests in derivation check phases;
- language unit and integration tests as derivations;
- NixOS VM tests for system behavior;
- integrity hashes for vendored sources that formatters must not modify;
- policy checks for generated configuration and synchronized artifacts.

Network, hardware, secret-bearing, and destructive tests should be explicit apps. Document why they cannot be sandboxed and make their prerequisites fail clearly.

## Hooks

Pre-commit should remain fast enough that developers leave it enabled. Put complete tests and package builds in pre-push or the flake check. A custom hook command must use a Nix package path or a packaged application rather than an ambient executable.

## CI adapters

Replace the included workflow adapters when the project uses another Nix implementation, binary cache, remote builder, or CI service. Preserve the boundary: provisioning and cache setup may be provider-specific; ordinary project execution should remain the explicit one-line command `nix flake check --print-build-logs`.
