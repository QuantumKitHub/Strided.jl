module StridedAMDGPUExt

using Strided, StridedViews, AMDGPU
using AMDGPU: Adapt
using AMDGPU: GPUArrays

const ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

function Base.copy!(dst::StridedView{TD, ND, TAD, FD}, src::StridedView{TS, NS, TAS, FS}) where {TD <: Number, ND, TAD <: ROCArray{TD}, FD <: ALL_FS, TS <: Number, NS, TAS <: ROCArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS)
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

end
