module StridedcuBLASExt

using Strided, StridedViews, GPUArrays, cuBLAS, cuBLAS.CUDACore, LinearAlgebra
import Strided: blas_mul!, isblasmatrix

const CuStridedView{T, N, A <: CuArray{T}} = StridedViews.StridedView{T, N, A}

# A view of the `(lda, n)` matrix keeps the column stride as the leading dimension. The last
# column might not span `lda` elements in the parent, but `gemm!` only accesses `1:m`.
function _stridedstorage(A::CuStridedView{T, 2}) where {T}
    m, n = size(A)
    lda = max(stride(A, 2), m, 1)
    return view(GPUArrays.derive(T, parent(A), (lda, n), A.offset), 1:m, :)
end

function Strided.blas_mul!(C::CuStridedView{T, 2}, A::CuStridedView{T, 2}, B::CuStridedView{T, 2}, α::Number, β::Number) where {T <: LinearAlgebra.BlasFloat}
    isblasmatrix(C) || return Strided.__mul!(C, A, B, α, β)
    A2, CA = Strided.getblasmatrix(A)
    B2, CB = Strided.getblasmatrix(B)
    cuBLAS.gemm!(CA, CB, convert(T, α), _stridedstorage(A2), _stridedstorage(B2), convert(T, β), _stridedstorage(C))
    return C
end

end
