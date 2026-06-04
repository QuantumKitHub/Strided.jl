module StridedAMDGPUExt

using Strided, StridedViews, AMDGPU, AMDGPU.rocBLAS, LinearAlgebra
import Strided: blas_mul!

const ROCStridedView{T, N, A <: ROCArray{T}} = StridedViews.StridedView{T, N, A}

function Strided.blas_mul!(C::ROCStridedView{T, 2}, A::ROCStridedView{T, 2}, B::ROCStridedView{T, 2}, α::Number, β::Number) where {T <: LinearAlgebra.BlasFloat}
    A2, CA = Strided.getblasmatrix(A)
    B2, CB = Strided.getblasmatrix(B)
    C2, CC = Strided.getblasmatrix(C)
    A2a = Base.unsafe_wrap(ROCMatrix{T}, pointer(A2), size(A2))
    B2a = Base.unsafe_wrap(ROCMatrix{T}, pointer(B2), size(B2))
    C2a = Base.unsafe_wrap(ROCMatrix{T}, pointer(C2), size(C2))
    AMDGPU.rocBLAS.gemm!(CA, CB, convert(T, α), A2a, B2a, convert(T, β), C2a)
    return C
end

end
