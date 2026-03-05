module StridedGPUArraysExt

using Strided, GPUArrays, LinearAlgebra
using GPUArrays: Adapt, KernelAbstractions
using GPUArrays.KernelAbstractions: @kernel, @index

ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

KernelAbstractions.get_backend(sv::StridedView{T, N, TA}) where {T, N, TA <: AnyGPUArray{T}} = KernelAbstractions.get_backend(parent(sv))

function Base.Broadcast.BroadcastStyle(gpu_sv::StridedView{T, N, TA}) where {T, N, TA <: AnyGPUArray{T}}
    raw_style = Base.Broadcast.BroadcastStyle(TA)
    return typeof(raw_style)(Val(N)) # sets the dimensionality correctly
end

function Base.copy!(dst::AbstractArray{TD, ND}, src::StridedView{TS, NS, TAS, FS}) where {TD <: Number, ND, TS <: Number, NS, TAS <: AbstractGPUArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS)
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

# lifted from GPUArrays.jl
function Base.fill!(A::StridedView{T, N, TA, F}, x) where {T, N, TA <: AbstractGPUArray{T}, F <: ALL_FS}
    isempty(A) && return A
    @kernel function fill_kernel!(a, val)
        idx = @index(Global, Linear)
        @inbounds a[idx] = val
    end
    # ndims check for 0D support
    kernel = fill_kernel!(KernelAbstractions.get_backend(A))
    kernel(A, x; ndrange = length(A))
    return A
end

function LinearAlgebra.mul!(
        C::StridedView{TC, 2, <:AnyGPUArray{TC}},
        A::StridedView{TA, 2, <:AnyGPUArray{TA}}, B::AnyGPUMatrix{TB},
        α::Number = true, β::Number = false
    ) where {TA, TB, TC}
    return mul!(C, A, StridedView(B), α, β)
end

function Strided.__mul!(
        C::StridedView{TC, 2, <:AnyGPUArray{TC}},
        A::StridedView{TA, 2, <:AnyGPUArray{TA}},
        B::StridedView{TB, 2, <:AnyGPUArray{TB}},
        α::Number, β::Number
    ) where {TC, TA, TB}
    return GPUArrays.generic_matmatmul!(C, A, B, α, β)
end

end
