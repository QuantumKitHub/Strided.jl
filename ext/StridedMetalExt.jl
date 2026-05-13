module StridedMetalExt

using Strided, Metal, GPUArrays, LinearAlgebra
using Metal: MtlArray

const MtlSV{T} = StridedView{T, 2, <:MtlArray{T}}

# Metal MPS/MPSGraph don't support non-unit leading dimensions, so only contiguous
# matrices (stride(A,2) == size(A,1)) are handled here; others fall back to __mul!.
_iscontiguous(A::StridedView{T, 2}) where {T} = stride(A, 2) == size(A, 1)

function Strided.blas_mul!(tA::AbstractChar, tB::AbstractChar, α::T,
        A::MtlSV{T}, B::MtlSV{T}, β::T, C::MtlSV{T}) where {T <: LinearAlgebra.BlasFloat}
    _iscontiguous(A) && _iscontiguous(B) && _iscontiguous(C) ||
        return Strided.__mul!(C, A, B, α, β)
    # Reconstruct 2D MtlArray views pointing to the same GPU memory.
    # GPUArrays.derive handles the buffer + offset correctly.
    A2 = GPUArrays.derive(T, parent(A), size(A), A.offset)
    B2 = GPUArrays.derive(T, parent(B), size(B), B.offset)
    C2 = GPUArrays.derive(T, parent(C), size(C), C.offset)
    wA = tA == 'N' ? A2 : (tA == 'T' ? transpose(A2) : adjoint(A2))
    wB = tB == 'N' ? B2 : (tB == 'T' ? transpose(B2) : adjoint(B2))
    LinearAlgebra.mul!(C2, wA, wB, α, β)
end

end
