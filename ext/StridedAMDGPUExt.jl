module StridedAMDGPUExt

using Strided, AMDGPU, LinearAlgebra
using AMDGPU: rocBLAS, ROCArray

const ROCSV{T} = StridedView{T, 2, <:ROCArray{T}}

# Use the low-level pointer + explicit (m, n, k, lda) rocBLAS overload so that
# strided slices (lda > m) work the same way as CPU BLAS.
# pointer(sv::StridedView) returns a ROCPtr{T} with the correct element offset applied.
function Strided.blas_mul!(tA::AbstractChar, tB::AbstractChar, α::T,
        A::ROCSV{T}, B::ROCSV{T}, β::T, C::ROCSV{T}) where {T <: LinearAlgebra.BlasFloat}
    m, n = size(C)
    k = tA == 'N' ? size(A, 2) : size(A, 1)
    rocBLAS.gemm!(tA, tB, m, n, k,
        α, pointer(A), stride(A, 2),
        pointer(B), stride(B, 2),
        β, pointer(C), stride(C, 2))
end

end
