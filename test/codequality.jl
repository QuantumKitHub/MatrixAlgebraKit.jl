using MatrixAlgebraKit

using Aqua
Aqua.test_all(MatrixAlgebraKit)

using JET
JET.test_package(MatrixAlgebraKit; target_defined_modules = true)
