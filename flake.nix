{
  description = "Rust template with binary, library, Fenix, and Crane";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    crane = {
      url = "github:ipetkov/crane";
    };
  };

  outputs = {
    self,
    nixpkgs,
    fenix,
    crane,
  }: let
    lib = nixpkgs.lib;
    systems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = lib.genAttrs systems;
  in {
    devShells = forAllSystems (system: let
      overlays = [fenix.overlays.default];
      pkgs = import nixpkgs {
        inherit system overlays;
      };

      rustToolchain = pkgs.fenix.fromToolchainFile {
        file = ./rust-toolchain.toml;
        sha256 = "mvUGEOHYJpn3ikC5hckneuGixaC+yGrkMM/liDIDgoU=";
      };

      craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
      src = craneLib.cleanCargoSource ./.;

      commonArgs = {
        inherit src;
        strictDeps = true;
      };
    in {
      default = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          pkg-config
        ];

        buildInputs = with pkgs; [
          openblasCompat
          gfortran.cc.lib
        ];

        packages = with pkgs; [
          rustToolchain

          cargo-nextest
          cargo-edit
          cargo-llvm-cov
          cargo-readme
          cargo-release
          release-plz
          cargo-watch
          git-cliff
          cargo-deny
          cargo-udeps
          cargo-semver-checks
          just

          pkg-config
          fontconfig
          openssl
          gcc
          gdb
          lldb
        ];

        LD_LIBRARY_PATH = lib.makeLibraryPath [
          pkgs.openblasCompat
          pkgs.gfortran.cc.lib
        ];

        shellHook = ''
          echo "Rust environment ready"
          echo
          echo "Common commands:"
          echo "  just run"
          echo "  just test"
          echo "  just nextest"
          echo "  just coverage"
          echo "  just lint"
          echo "  just fmt"
          echo "  nix build"
          echo "  nix flake check"
        '';
      };
    });

    packages = forAllSystems (system: let
      overlays = [fenix.overlays.default];
      pkgs = import nixpkgs {
        inherit system overlays;
      };

      commonArgs = {
        inherit src;
        strictDeps = true;

        nativeBuildInputs = with pkgs; [
          pkg-config
        ];

        buildInputs = with pkgs; [
          openblas
          gfortran.cc.lib
        ];
      };

      rustToolchain = pkgs.fenix.fromToolchainFile {
        file = ./rust-toolchain.toml;
        sha256 = "mvUGEOHYJpn3ikC5hckneuGixaC+yGrkMM/liDIDgoU=";
      };

      craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;
      src = craneLib.cleanCargoSource ./.;

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      myCrate = craneLib.buildPackage (commonArgs
        // {
          inherit src cargoArtifacts;
          strictDeps = true;
        });
    in {
      default = myCrate;
    });

    checks = forAllSystems (system: let
      overlays = [fenix.overlays.default];
      pkgs = import nixpkgs {
        inherit system overlays;
      };

      commonArgs = {
        inherit src;
        strictDeps = true;

        nativeBuildInputs = with pkgs; [
          pkg-config
        ];

        buildInputs = with pkgs; [
          fontconfig
        ];
      };

      rustToolchain = pkgs.fenix.fromToolchainFile {
        file = ./rust-toolchain.toml;
        sha256 = "mvUGEOHYJpn3ikC5hckneuGixaC+yGrkMM/liDIDgoU=";
      };

      craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

      src = craneLib.cleanCargoSource ./.;

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;
    in {
      rust-build = craneLib.buildPackage (commonArgs
        // {
          inherit src cargoArtifacts;
          strictDeps = true;
        });

      rust-clippy = craneLib.cargoClippy (commonArgs
        // {
          inherit src cargoArtifacts;
          cargoClippyExtraArgs = "--all-targets --all-features -- -D warnings";
        });

      rust-doc = craneLib.cargoDoc (commonArgs
        // {
          inherit src cargoArtifacts;
        });

      rust-fmt = craneLib.cargoFmt {
        inherit src;
      };

      rust-test = craneLib.cargoTest (commonArgs
        // {
          inherit src cargoArtifacts;
        });
    });
  };
}
