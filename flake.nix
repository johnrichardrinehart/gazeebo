{
  description = "Local gaze-driven cursor navigation for Linux desktops";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{
      self,
      flake-parts,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      imports = [ inputs.flake-parts.flakeModules.partitions ];

      # Authoring inputs stay out of package consumers' input closures.
      partitionedAttrs = {
        checks = "dev";
        devShells = "dev";
        formatter = "dev";
      };

      partitions.dev = {
        extraInputsFlake = ./dev;
        module = import ./nix/flake/dev-partition.nix;
      };

      perSystem =
        { pkgs, ... }:
        let
          packages = import ./nix/packages {
            inherit pkgs;
            revision = self.rev or self.dirtyRev or "dirty";
          };
        in
        {
          inherit packages;

          apps.default = {
            type = "app";
            program = "${packages.gazeebo}/bin/gazeebo";
            meta.description = "Run process-scoped gaze-driven cursor navigation";
          };
        };
    };
}
