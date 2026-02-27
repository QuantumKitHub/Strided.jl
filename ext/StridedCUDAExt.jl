module StridedCUDAExt

using Strided, CUDA
using CUDA: Adapt, CuPtr, KernelAdaptor
using CUDA: GPUArrays

const ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

const CuStridedView{T, N, A <: CuArray{T}} = StridedView{T, N, A}

function Adapt.adapt_structure(to, A::CuStridedView)
    return StridedView(
        Adapt.adapt_structure(to, parent(A)),
        A.size, A.strides, A.offset, A.op
    )
end

function Base.pointer(x::CuStridedView{T}) where {T}
    return Base.unsafe_convert(CuPtr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{CuPtr{T}}, a::CuStridedView{T}) where {T}
    return convert(CuPtr{T}, pointer(a))
end

function Base.print_array(io::IO, X::CuStridedView)
    return Base.print_array(io, Adapt.adapt_structure(Array, X))
end

function Base.copy!(dst::StridedView{TD, ND, TAD, FD}, src::StridedView{TS, NS, TAS, FS}) where {TD <: Number, ND, TAD <: CuArray{TD}, FD <: ALL_FS, TS <: Number, NS, TAS <: CuArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS)
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

end
