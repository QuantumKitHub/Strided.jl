module StridedGPUArraysExt

using Strided, GPUArrays
using GPUArrays: Adapt, KernelAbstractions

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

end
