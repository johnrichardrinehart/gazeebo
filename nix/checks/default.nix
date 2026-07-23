{
  pkgs,
  self,
  system,
}:
let
  source = pkgs.lib.fileset.toSource {
    root = ../..;
    fileset = pkgs.lib.fileset.unions [
      ../../pyproject.toml
      ../../src
      ../../tests
    ];
  };
  python = pkgs.python3.withPackages (pythonPackages: [
    pythonPackages.dbus-next
    pythonPackages.mypy
    pythonPackages.numpy
  ]);
in
{
  package = self.packages.${system}.default;

  typecheck = pkgs.runCommand "gazeebo-typecheck" { nativeBuildInputs = [ python ]; } ''
    cp -r ${source} source
    chmod -R u+w source
    cd source
    mypy src tests
    touch $out
  '';
}
