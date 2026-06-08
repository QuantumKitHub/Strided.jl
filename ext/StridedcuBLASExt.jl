module StridedcuBLASExt

using Strided, StridedViews, cuBLAS, cuBLAS.CUDACore, LinearAlgebra
import Strided: blas_mul!

const CuStridedView{T, N, A <: CuArray{T}} = StridedViews.StridedView{T, N, A}

function Strided.blas_mul!(C::CuStridedView{T, 2}, A::CuStridedView{T, 2}, B::CuStridedView{T, 2}, α::Number, β::Number) where {T <: LinearAlgebra.BlasFloat}
    A2, CA = Strided.getblasmatrix(A)
    B2, CB = Strided.getblasmatrix(B)
    C2, CC = Strided.getblasmatrix(C)
    A2a = Base.unsafe_wrap(CuMatrix{T}, pointer(A2), size(A2))
    B2a = Base.unsafe_wrap(CuMatrix{T}, pointer(B2), size(B2))
    C2a = Base.unsafe_wrap(CuMatrix{T}, pointer(C2), size(C2))
    cuBLAS.gemm!(CA, CB, convert(T, α), A2a, B2a, convert(T, β), C2a)
    return C
end

end
