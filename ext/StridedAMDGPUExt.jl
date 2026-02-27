module StridedAMDGPUExt

using Strided, AMDGPU
using AMDGPU: Adapt
using AMDGPU: GPUArrays

const ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

const ROCStridedView{T, N, A <: ROCArray{T}} = StridedView{T, N, A}

function Adapt.adapt_structure(to, A::ROCStridedView)
    return StridedView(
        Adapt.adapt_structure(to, parent(A)),
        A.size, A.strides, A.offset, A.op
    )
end

function Base.pointer(x::ROCStridedView{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, pointer(x.parent, x.offset + 1))
end
function Base.unsafe_convert(::Type{Ptr{T}}, a::ROCStridedView{T}) where {T}
    return convert(Ptr{T}, pointer(a))
end

function Base.print_array(io::IO, X::ROCStridedView)
    return Base.print_array(io, Adapt.adapt_structure(Array, X))
end

function Base.copy!(dst::StridedView{TD, ND, TAD, FD}, src::StridedView{TS, NS, TAS, FS}) where {TD <: Number, ND, TAD <: ROCArray{TD}, FD <: ALL_FS, TS <: Number, NS, TAS <: ROCArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS)
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

end
