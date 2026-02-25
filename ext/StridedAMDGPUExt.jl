module StridedAMDGPUExt

using Strided, AMDGPU
using AMDGPU: Adapt, Runtime.Adaptor
using AMDGPU: GPUArrays

const ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

function Adapt.adapt_storage(to::Runtime.Adaptor, xs::StridedView{T,N,TA,F}) where {T,N,TA<:ROCArray{T},F <: ALL_FS}
    return StridedView(Adapt.adapt(to, parent(xs)), xs.size, xs.strides, xs.offset, xs.op)
end

function Base.copy!(dst::StridedView{TD, ND, TAD, FD}, src::StridedView{TS, NS, TAS, FS}) where {TD, ND, TAD <: ROCArray{TD}, FD <: ALL_FS, TS, NS, TAS <: ROCArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS) 
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

end
