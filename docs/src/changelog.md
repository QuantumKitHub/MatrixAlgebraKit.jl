# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Guidelines for updating this changelog

When making changes to this project, please update the "Unreleased" section with your changes under the appropriate category:

- **Added** for new features.
- **Changed** for changes in existing functionality.
- **Deprecated** for soon-to-be removed features.
- **Removed** for now removed features.
- **Fixed** for any bug fixes.
- **Performance** for performance improvements.

When releasing a new version, move the "Unreleased" changes to a new version section with the release date.

## [Unreleased](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/compare/v0.6.0...HEAD)

### Added

### Changed

### Deprecated

### Removed

### Fixed

### Security

## [0.6.0](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/releases/tag/v0.6.0) - 2025-11-14

### Added
- New `project_isometric` function for projecting matrices onto isometric manifold ([#67](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/67))
- New `PolarNewton` algorithm for polar decomposition ([#67](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/67))
- New matrix property functions: `ishermitian`, `isantihermitian`, `hermitianpart!`, `hermitianpart`, `antihermitianpart!`, and `antihermitianpart` ([#64](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/64))
- Support for `BigFloat` via new `GenericLinearAlgebra` extension ([#87](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/87))
- Mooncake reverse-mode AD rules ([#85](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/85))
- GPU support for image and null space computations ([#82](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/82))
- GPU support for polar decomposition ([#83](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/83))
- GPU support for new projection operations ([#81](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/81))
- Output truncation error for truncated decompositions ([#75](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/75))
- Documentation for truncated decomposition keyword arguments ([#71](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/71))
- Default algorithm implementations for GPU wrapper array types ([#49](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/49))

### Changed

- Made `gaugefix!` optional ([#95](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/95))
- Renamed `isisometry` to `isisometric` for consistency with `project_isometric` ([#73](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/73))
- Refactored `left_orth`, `right_orth`, `left_null` and `right_null` interface ([#79](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/79))
- Improved GPU support for SVD operations ([#80](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/80))
- Loosened strictness on hermitian checks ([#78](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/78))
- Updated pullback tolerances ([#92](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/pull/92))

### Removed

### Fixed
