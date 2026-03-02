module StridedJLArraysExt

using Strided, StridedViews, JLArrays
using JLArrays: Adapt
using JLArrays: GPUArrays

const ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

function Base.copy!(dst::StridedView{TD, ND, TAD, FD}, src::StridedView{TS, NS, TAS, FS}) where {TD <: Number, ND, TAD <: JLArray{TD}, FD <: ALL_FS, TS <: Number, NS, TAS <: JLArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS)
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

function Base.copy!(dst::AbstractArray{TD, ND}, src::StridedView{TS, NS, TAS, FS}) where {TD <: Number, ND, TS <: Number, NS, TAS <: JLArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS)
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

end
