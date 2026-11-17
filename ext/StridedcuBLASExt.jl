module StridedcuBLASExt

using Strided, StridedViews, GPUArrays, cuBLAS, cuBLAS.CUDACore, LinearAlgebra
import Strided: blas_mul!, isblasmatrix, getblasmatrix

const CuStridedView{T, N, A <: CuArray{T}} = StridedViews.StridedView{T, N, A}

function Strided.blas_mul!(C::CuStridedView{T, 2}, A::CuStridedView{T, 2}, B::CuStridedView{T, 2}, α::Number, β::Number) where {T <: LinearAlgebra.BlasFloat}
    A2, CA = getblasmatrix(A)
    B2, CB = getblasmatrix(B)
    C2, _ = getblasmatrix(C)
    cuBLAS.gemm!(CA, CB, convert(T, α), A2, B2, convert(T, β), C2)
    return C
end

end
