# Secret contracts

SecretSpec separates the names and requirements expected by a project from the provider that stores their values.

## Project policy

- Commit `secretspec.toml`; never commit resolved values.
- Do not read secrets during flake evaluation.
- Do not place secrets in derivation arguments, because derivations and the Nix store are not secret.
- Do not inject all project secrets into the default development shell.
- Resolve secrets at the narrowest runtime command boundary.
- Prefer `as_path = true` for certificates, private keys, and credential documents.
- Require an access reason for coding agents unless a project documents why that audit control is unsuitable.
- Give CI and deployment identities only the provider permissions needed for their profile.

Copy `nix/examples/secretspec.toml` to the repository root when the project first needs secrets, then replace the example declaration.

## Profiles and providers

A declaration can keep stable application-facing names while profiles select different requirements and providers:

- local development: keyring, `pass`, `gopass`, or a password manager;
- CI: environment variables supplied by the runner's secret facility;
- production: Vault/OpenBao or a cloud secret manager.

Use `dotenv` only when its plaintext-at-rest and file-lifecycle properties are acceptable. Keep dotenv files ignored.

## Runtime invocation

Expose secrets only to the target process:

```console
secretspec check --profile production
secretspec run --profile production -- nix run .#deploy
```

An application with a supported SecretSpec SDK can resolve a typed secret contract directly and avoid broad environment injection.

## Deployment-time secret files

SecretSpec and host secret managers solve different boundaries. sops-nix or agenix remain appropriate when NixOS activation must materialize a secret file for a systemd service. SecretSpec is appropriate for declaring portable runtime requirements and resolving them from environment-specific providers.

Do not convert a file-based deployment secret into a long-lived shell variable merely to make every secret use the same transport.
