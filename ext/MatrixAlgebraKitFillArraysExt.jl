module MatrixAlgebraKitFillArraysExt

using LinearAlgebra
using MatrixAlgebraKit
using MatrixAlgebraKit: AbstractAlgorithm
using FillArrays
using FillArrays: AbstractZerosMatrix

struct EyeAlgorithm <: AbstractAlgorithm end
struct ZerosAlgorithm <: AbstractAlgorithm end

for f in [:eig_full,
          :eigh_full,
          :qr_compact,
          :qr_full,
          :left_polar,
          :lq_compact,
          :lq_full,
          :right_polar,
          :svd_compact,
          :svd_full]
    @eval begin
        MatrixAlgebraKit.copy_input(::typeof($f), a::Eye) = a

        MatrixAlgebraKit.copy_input(::typeof($f), a::AbstractZerosMatrix) = a
    end
end

for f in [:eig, :eigh, :lq, :qr, :polar, :svd]
    ff = Symbol("default_", f, "_algorithm")
    @eval begin
        function MatrixAlgebraKit.$ff(a::Type{<:Eye}; kwargs...)
            return EyeAlgorithm()
        end

        function MatrixAlgebraKit.$ff(a::Type{<:AbstractZerosMatrix}; kwargs...)
            return ZerosAlgorithm()
        end
    end
end

for f in [:eig_full!,
          :eigh_full!,
          :qr_compact!,
          :qr_full!,
          :left_polar!,
          :lq_compact!,
          :lq_full!,
          :right_polar!]
    @eval begin
        nfactors(::typeof($f)) = 2
    end
end
for f in [:svd_compact!, :svd_full!]
    @eval begin
        nfactors(::typeof($f)) = 3
    end
end

for f in [:eig_full!,
          :eigh_full!,
          :qr_compact!,
          :qr_full!,
          :left_polar!,
          :lq_compact!,
          :lq_full!,
          :right_polar!,
          :svd_compact!,
          :svd_full!]
    @eval begin
        function MatrixAlgebraKit.initialize_output(::typeof($f), a::Eye,
                                                    alg::EyeAlgorithm)
            return nothing
        end
        function MatrixAlgebraKit.check_input(::typeof($f), A::Eye, F)
            LinearAlgebra.checksquare(A)
            return nothing
        end

        function MatrixAlgebraKit.$f(a::Eye, F, alg::EyeAlgorithm; kwargs...)
            return ntuple(_ -> a, nfactors($f))
        end

        function MatrixAlgebraKit.initialize_output(::typeof($f), a::AbstractZerosMatrix,
                                                    alg::ZerosAlgorithm)
            return nothing
        end
        function MatrixAlgebraKit.check_input(::typeof($f), A::AbstractZerosMatrix, F)
            LinearAlgebra.checksquare(A)
            return nothing
        end
        function MatrixAlgebraKit.$f(a::AbstractZerosMatrix, F, alg::ZerosAlgorithm;
                                     kwargs...)
            return ntuple(_ -> a, nfactors($f))
        end
    end
end

end
