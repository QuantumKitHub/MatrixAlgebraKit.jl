```@meta
CurrentModule = MatrixAlgebraKit
CollapsedDocStrings = true
```

# Matrix Properties

MatrixAlgebraKit.jl provides a number of methods to check various properties of matrices.

```@docs; canonical=false
isisometry
isunitary
ishermitian
isantihermitian
```

Furthermore, there are also methods to project a matrix onto the nearest matrix (in 2-norm distance) with a given property.

```@docs; canonical=false
project_isometric
project_hermitian
project_antihermitian
```