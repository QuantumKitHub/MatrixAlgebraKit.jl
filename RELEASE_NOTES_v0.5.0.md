## MatrixAlgebraKit v0.5.0

[Diff since v0.4.1](https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/compare/v0.4.1...v0.5.0)

This release introduces pullback functions for nullspace decompositions, refactors the truncation interface to return indices, and includes several code quality improvements and internal consistency enhancements.

---

## 🚀 New Features & Enhancements

- **Nullspace Pullback Functions**: Added dedicated pullback functions `qr_null_pullback!` and `lq_null_pullback!` for computing gradients through nullspace decompositions. These functions provide a cleaner and more efficient interface for automatic differentiation through `qr_null` and `lq_null` operations.

- **Renamed Pullback Functions**: The pullback functions have been renamed for consistency:
  - `qr_compact_pullback!` → `qr_pullback!`
  - `lq_compact_pullback!` → `lq_pullback!`
  
  The new names are marked as `public` and are now part of the package's public API.

- **Improved Truncation Interface**: The internal `truncate` function now returns both the truncated result and the indices that were kept, enabling more efficient implementations in ChainRules extensions and other downstream code.

- **Enhanced ChainRules Integration**: 
  - Streamlined pullback implementations for `qr_null!` and `lq_null!` to use the new dedicated pullback functions
  - Improved type stability in truncation-related rules
  - Added `@non_differentiable` declarations for utility functions (`select_algorithm`, `initialize_output`, `check_input`, `isisometry`, `isunitary`)
  - More generic type signatures (removed `AbstractMatrix` constraints) for better compatibility with custom array types

---

## 🛠️ Internal & Code Quality

- **Refactored Truncation Implementation**: Changed `truncate!` to `truncate` (returning new values rather than mutating), with the signature now returning `(result, indices)` instead of just `result`. This change is internal and does not affect the public API.

- **Documentation Updates**: Updated internal documentation references from `truncate!` to `truncate`.

- **Code Consistency**: Various internal code improvements for better consistency and maintainability:
  - Cleaned up implementation files to remove unnecessary exports
  - Improved formatting consistency across pullback functions
  - Simplified some conditional logic in implementations

---

## 📋 Public API Changes

### New Public Functions
- `qr_null_pullback!` - Compute pullback for QR nullspace decomposition
- `lq_null_pullback!` - Compute pullback for LQ nullspace decomposition
- `truncate` - Now marked as public (previously internal)

### Renamed Public Functions
- `qr_compact_pullback!` → `qr_pullback!` (old name removed)
- `lq_compact_pullback!` → `lq_pullback!` (old name removed)

**Note**: The previous pullback function names are no longer available. If you were using these internal functions (they were not exported but marked as public in v0.4.1), you will need to update your code to use the new names.

---

## 🔧 Migration Guide

If you were using the pullback functions directly:

```julia
# Old code (v0.4.1)
qr_compact_pullback!(ΔA, A, QR, ΔQR)
lq_compact_pullback!(ΔA, A, LQ, ΔLQ)

# New code (v0.5.0)
qr_pullback!(ΔA, A, QR, ΔQR)
lq_pullback!(ΔA, A, LQ, ΔLQ)
```

For nullspace pullbacks, use the new dedicated functions:

```julia
# v0.5.0
qr_null_pullback!(ΔA, A, N, ΔN)
lq_null_pullback!(ΔA, A, Nᴴ, ΔNᴴ)
```

---

For further details, see the updated docs and the diff for this release: https://github.com/QuantumKitHub/MatrixAlgebraKit.jl/compare/v0.4.1...v0.5.0

**Merged pull requests:**
- Add `qr_null_pullback!` and `lq_null_pullback!` (#62) (@lkdvos)
