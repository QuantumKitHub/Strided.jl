module StridedCUDAExt

using Strided, CUDA
using CUDA: Adapt, KernelAdaptor
using CUDA: GPUArrays

const ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

function Adapt.adapt_storage(to::KernelAdaptor, xs::StridedView{T,N,TA,F}) where {T,N,TA<:CuArray{T},F <: ALL_FS}
    return StridedView(Adapt.adapt(to, parent(xs)), xs.size, xs.strides, xs.offset, xs.op)
end

function Base.copy!(dst::StridedView{TD, ND, TAD, FD}, src::StridedView{TS, NS, TAS, FS}) where {TD, ND, TAD <: CuArray{TD}, FD <: ALL_FS, TS, NS, TAS <: CuArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS) 
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

end
