module StridedcuBLASExt

using Strided, StridedViews, GPUArrays, cuBLAS, cuBLAS.CUDACore, LinearAlgebra
import Strided: blas_mul!, isblasmatrix, getblasmatrix

const CuStridedView{T, N, A <: CuArray{T}} = StridedViews.StridedView{T, N, A}

# A view of the `(lda, n)` matrix keeps the column stride as the leading dimension. The last
# column might not span `lda` elements in the parent, but `gemm!` only accesses `1:m`.
function Strided.getblasmatrix(A::CuStridedView{T, 2}) where {T <: LinearAlgebra.BlasFloat}
    A2, CA = Strided._blasview(A)
    m, n = size(A2)
    lda = max(stride(A2, 2), m, 1)
    return view(GPUArrays.derive(T, parent(A2), (lda, n), A2.offset), 1:m, :), CA
end

function Strided.blas_mul!(C::CuStridedView{T, 2}, A::CuStridedView{T, 2}, B::CuStridedView{T, 2}, α::Number, β::Number) where {T <: LinearAlgebra.BlasFloat}
    isblasmatrix(C) || return Strided.__mul!(C, A, B, α, β)
    A2, CA = getblasmatrix(A)
    B2, CB = getblasmatrix(B)
    C2, _ = getblasmatrix(C) # `_mul!` guarantees `C.op == identity && stride(C, 1) < stride(C, 2)`
    cuBLAS.gemm!(CA, CB, convert(T, α), A2, B2, convert(T, β), C2)
    return C
end

end
