# MatrixAlgebraKit

A Julia interface for matrix algebra, with a focus on performance, flexibility and extensibility.

| **Documentation** | **Build Status** | **Coverage** | **Quality assurance** |
|:-----------------:|:----------------:|:------------:|:---------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-dev-img]][docs-dev-url] | [![CI][ci-img]][ci-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] [![JET QA][jet-img]][jet-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://QuantumKitHub.github.io/MatrixAlgebraKit.jl/stable

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://QuantumKitHub.github.io/MatrixAlgebraKit.jl/dev

[ci-img]: https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/actions/workflows/Tests.yml/badge.svg
[ci-url]: https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/actions/workflows/Tests.yml

[codecov-img]: https://codecov.io/gh/QuantumKitHub/MatrixAlgebraKit.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/QuantumKitHub/MatrixAlgebraKit.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[jet-img]: https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a
[jet-url]: https://github.com/aviatesk/JET.jl

This package provides an alternative interface to some of the matrix algebra functionality provided by the 
`LinearAlgebra` standard library.

The main goals of this package are:
* Definition of a common interface that is sufficiently expressive to allow easy adoption and extension.
* Ability to pass pre-allocated output arrays where the result of a computation is stored.
* Ability to easily switch between different backends and algorithms for the same operation.
* First class availability of pullback rules that can be used in combination with different AD ecosystems.
