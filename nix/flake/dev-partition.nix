{ inputs, self, ... }:
{
  imports = [
    inputs.git-hooks.flakeModule
    inputs.treefmt-nix.flakeModule
  ];

  perSystem =
    {
      config,
      pkgs,
      system,
      ...
    }:
    {
      treefmt = {
        projectRootFile = "flake.nix";
        flakeCheck = false;
        programs = {
          actionlint.enable = true;
          clang-format.enable = true;
          deadnix.enable = true;
          nixfmt.enable = true;
          prettier.enable = true;
          ruff-check.enable = true;
          ruff-format.enable = true;
          shellcheck.enable = true;
          shfmt.enable = true;
          statix.enable = true;
          taplo.enable = true;
        };
        settings.formatter = {
          shellcheck.includes = [
            "*.sh"
            ".envrc"
          ];
          shfmt.includes = [
            "*.sh"
            ".envrc"
          ];
        };
      };

      pre-commit.settings.hooks = {
        treefmt.enable = true;
        check-added-large-files.enable = true;
        check-merge-conflicts.enable = true;

        flake-check-before-push = {
          enable = true;
          name = "authoritative Nix checks";
          entry = "${pkgs.nix}/bin/nix flake check --print-build-logs";
          always_run = true;
          pass_filenames = false;
          require_serial = true;
          stages = [ "pre-push" ];
        };
      };

      checks = import ../checks {
        inherit pkgs self system;
      };

      devShells.default = pkgs.mkShell {
        packages = [
          pkgs.nixd
          pkgs.python3Packages.mypy
          pkgs.python3Packages.ruff
          self.packages.${system}.default
        ];
        shellHook = config.pre-commit.installationScript;
      };
    };
}
