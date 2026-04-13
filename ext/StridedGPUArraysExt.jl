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

function Base.copyto!(dest::StridedView{T, N, <:AnyGPUArray{T}}, bc::Base.Broadcast.Broadcasted{Strided.StridedArrayStyle{N}}) where {T <: Number, N}
    dims = size(dest)
    any(isequal(0), dims) && return dest

    GPUArrays._copyto!(dest, bc)
    return dest
end

# lifted from GPUArrays.jl
function Base.fill!(A::StridedView{T, N, TA, F}, x) where {T, N, TA <: AbstractGPUArray{T}, F <: ALL_FS}
    isempty(A) && return A
    @kernel function fill_kernel!(a, val)
        idx = @index(Global, Cartesian)
        @inbounds a[idx] = val
    end
    # ndims check for 0D support
    kernel = fill_kernel!(KernelAbstractions.get_backend(A))
    f_x = F <: Union{typeof(conj), typeof(adjoint)} ? conj(x) : x
    kernel(A, f_x; ndrange = size(A))
    return A
end

function LinearAlgebra.mul!(
        C::StridedView{TC, 2, <:AnyGPUArray{TC}},
        A::StridedView{TA, 2, <:AnyGPUArray{TA}},
        B::StridedView{TB, 2, <:AnyGPUArray{TB}},
        α::Number, β::Number
    ) where {TC, TA, TB}
    return GPUArrays.generic_matmatmul!(C, A, B, α, β)
end

end
