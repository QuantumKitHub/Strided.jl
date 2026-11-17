module StridedAMDGPUExt

using Strided, StridedViews, GPUArrays, AMDGPU, AMDGPU.rocBLAS, LinearAlgebra
import Strided: blas_mul!, isblasmatrix, getblasmatrix

const ROCStridedView{T, N, A <: ROCArray{T}} = StridedViews.StridedView{T, N, A}

function Strided.blas_mul!(C::ROCStridedView{T, 2}, A::ROCStridedView{T, 2}, B::ROCStridedView{T, 2}, α::Number, β::Number) where {T <: LinearAlgebra.BlasFloat}
    isblasmatrix(C) || return Strided.__mul!(C, A, B, α, β)
    A2, CA = getblasmatrix(A)
    B2, CB = getblasmatrix(B)
    C2, _ = getblasmatrix(C)
    AMDGPU.rocBLAS.gemm!(CA, CB, convert(T, α), A2, B2, convert(T, β), C2)
    return C
end

end
