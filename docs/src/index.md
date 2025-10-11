# MatrixAlgebraKit.jl

*A Julia interface for matrix algebra, with a focus on performance, flexibility and extensibility.*

## Installation

MatrixAlgebraKit.jl is part of the general registry, and can be installed through the package manager:

```
pkg> add MatrixAlgebraKit
```

## Key features

The main goals of this package are:

* Definition of a common interface that is sufficiently expressive to allow easy adoption and extension.
* Ability to pass pre-allocated output arrays where the result of a computation is stored.
* Ability to easily switch between different backends and algorithms for the same operation.
* First class availability of pullback rules that can be used in combination with different AD ecosystems.

## User Interface

On the user-facing side of this package, we provide various implementations and interfaces for different matrix algebra operations.
These operations typically follow some common skeleton, and here we go into a little more detail to what behavior can be expected.

```@contents
Pages = ["user_interface/compositions.md", "user_interface/decompositions.md",
         "user_interface/truncations.md", "user_interface/properties.md",
         "user_interface/matrix_functions.md"]
Depth = 2
```
