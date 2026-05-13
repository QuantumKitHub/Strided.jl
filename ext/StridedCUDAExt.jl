module StridedCUDAExt

using Strided, CUDACore, LinearAlgebra
using CUDACore: CUBLAS, CuArray

const CuSV{T} = StridedView{T, 2, <:CuArray{T}}

# Use the low-level pointer + explicit (m, n, k, lda) CUBLAS overload so that
# strided slices (lda > m) work the same way as CPU BLAS.
# pointer(sv::StridedView) returns a CuPtr{T} with the correct element offset applied.
function Strided.blas_mul!(tA::AbstractChar, tB::AbstractChar, α::T,
        A::CuSV{T}, B::CuSV{T}, β::T, C::CuSV{T}) where {T <: LinearAlgebra.BlasFloat}
    m, n = size(C)
    k = tA == 'N' ? size(A, 2) : size(A, 1)
    CUBLAS.gemm!(tA, tB, m, n, k,
        α, pointer(A), stride(A, 2),
        pointer(B), stride(B, 2),
        β, pointer(C), stride(C, 2))
end

end
