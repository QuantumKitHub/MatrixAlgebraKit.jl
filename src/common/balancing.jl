"""
    balance!(A::AbstractMatrix{T}; radix=convert(T, 2)) where T

A pure Julia implementation of the GEBAL algorithm. 
Balances a matrix to improve eigenvalue accuracy.
"""
function balance!(A::AbstractMatrix{T}; radix=convert(T, 2)) where T
    n = LinearAlgebra.checksquare(A)
    
    scale = ones(real(T), n)
    low = 1
    high = n
    converged = false

    # --- Step 1: Permutation (Search for isolated eigenvalues) ---
    # (Simplified for brevity; most performance gain comes from scaling)
    # In a full GEBAL, we would swap rows/cols to move zeros to the corners.

    # --- Step 2: Scaling (The Ward Algorithm) ---
    while !converged
        converged = true
        for i in low:high
            # Calculate row and column norms (excluding diagonal)
            row_norm = 0.0
            col_norm = 0.0
            for j in low:high
                if i == j continue end
                row_norm += abs(A[i, j])
                col_norm += abs(A[j, i])
            end

            # Avoid division by zero
            if row_norm == 0 || col_norm == 0
                continue
            end

            # Iterative scaling to bring row and col norms closer
            g = row_norm / radix
            f = 1.0
            s = col_norm + row_norm
            
            # While column norm is significantly smaller than row norm
            while col_norm < g
                f *= radix
                col_norm *= radix^2
            end
            
            # While column norm is significantly larger than row norm
            g = row_norm * radix
            while col_norm >= g
                f /= radix
                col_norm /= radix^2
            end

            # Apply scaling if it's significant
            if (col_norm + row_norm) / f < 0.95 * s
                converged = false
                g = 1.0 / f
                scale[i] *= f
                # Apply to rows
                for j in 1:n
                    A[i, j] *= g
                end
                # Apply to columns
                for j in 1:n
                    A[j, i] *= f
                end
            end
        end
    end
    
    return A, scale
end


function permute_matrix!(A::AbstractMatrix{T}) where T
    n = size(A, 1)
    low = 1
    high = n
    
    # Simple implementation of the permutation search
    # We look for rows/cols that are essentially isolated
    
    # Search from the bottom up (high) and top down (low)
    changed = true
    while changed
        changed = false
        
        # Look for a column with only one non-zero element
        for j in high:-1:low
            col = A[low:high, j]
            if count(!iszero, col) == 1
                # Permute this column to the 'high' position
                swap_cols!(A, j, high)
                swap_rows!(A, j, high)
                high -= 1
                changed = true
            end
        end
        
        # Look for a row with only one non-zero element
        for i in low:high
            row = A[i, low:high]
            if count(!iszero, row) == 1
                # Permute this row to the 'low' position
                swap_cols!(A, i, low)
                swap_rows!(A, i, low)
                low += 1
                changed = true
            end
        end
    end
    return low, high
end

function swap_rows!(A, i, j)
    for k in axes(A, 2)
        A[i, k], A[j, k] = A[j, k], A[i, k]
    end
end
function swap_cols!(A, i, j)
    for k in axes(A, 1)
        A[k, i], A[k, j] = A[k, j], A[k, i]
    end
end