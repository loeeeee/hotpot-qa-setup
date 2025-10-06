{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = [
    pkgs.gcc-unwrapped.lib
    pkgs.python313
    pkgs.uv
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${pkgs.gcc-unwrapped.lib}/lib:$LD_LIBRARY_PATH"
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    # Activate uv virtual environment if it exists
    if [ -d ".venv" ]; then
      export PATH="$PWD/.venv/bin:$PATH"
    fi
  '';
}
