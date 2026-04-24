function isapproxone(A)
    return (size(A, 1) == size(A, 2)) && (A ≈ MatrixAlgebraKit.one!(similar(A)))
end
