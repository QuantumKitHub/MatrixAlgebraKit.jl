module LinearMaps

    export LinearMap

    using MatrixAlgebraKit
    using MatrixAlgebraKit: AbstractAlgorithm
    import MatrixAlgebraKit as MAK

    using LinearAlgebra: LinearAlgebra, lmul!, rmul!

    # Used to test non-AbstractMatrix codepaths.
    struct LinearMap{P <: AbstractMatrix}
        parent::P
    end
    Base.parent(A::LinearMap) = A.parent

    Base.copy!(dest::LinearMap, src::LinearMap) = (copy!(parent(dest), parent(src)); dest)

    # necessary for orth_svd default implementations
    LinearAlgebra.lmul!(D::LinearMap, A::LinearMap) = (lmul!(parent(D), parent(A)); A)
    LinearAlgebra.rmul!(A::LinearMap, D::LinearMap) = (rmul!(parent(A), parent(D)); A)
    LinearAlgebra.mul!(C::LinearMap, A::LinearMap, B::LinearMap, α, β) = LinearAlgebra.mul!(parent(C), parent(A), parent(B), α, β)

    for f in (:qr_compact, :lq_compact)
        @eval MAK.copy_input(::typeof($f), A::LinearMap) = LinearMap(MAK.copy_input($f, parent(A)))
    end

    for f! in (:qr_compact!, :lq_compact!, :svd_compact!, :svd_full!, :svd_trunc!)
        @eval MAK.check_input(::typeof($f!), A::LinearMap, F, alg::AbstractAlgorithm) =
            MAK.check_input($f!, parent(A), parent.(F), alg)
        @eval MAK.initialize_output(::typeof($f!), A::LinearMap, alg::AbstractAlgorithm) =
            LinearMap.(MAK.initialize_output($f!, parent(A), alg))
        @eval MAK.$f!(A::LinearMap, F, alg::AbstractAlgorithm) =
            LinearMap.(MAK.$f!(parent(A), parent.(F), alg))
    end

    for f in (:qr, :lq, :svd)
        default_f = Symbol(:default_, f, :_algorithm)
        @eval MAK.$default_f(::Type{LinearMap{A}}; kwargs...) where {A} = MAK.$default_f(A; kwargs...)
    end
end

using .LinearMaps
