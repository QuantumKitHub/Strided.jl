module StridedGPUArraysExt

using Strided, GPUArrays, LinearAlgebra
using GPUArrays: Adapt, KernelAbstractions
using GPUArrays.KernelAbstractions: @kernel, @index
using StridedViews: ParentIndex

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

# ---------- GPU mapreduce support ----------

@inline _gpu_init_acc(::Nothing, current_val) = current_val
@inline _gpu_init_acc(initop, current_val) = initop(current_val)

@inline _gpu_accum(::Nothing, acc, val) = val
@inline _gpu_accum(op, acc, val) = op(acc, val)

@inline function _strides_dot(strides::NTuple{N, Int}, cidx::CartesianIndex{N}) where {N}
    s = 0
    for d in Base.OneTo(N)
        @inbounds s += strides[d] * (cidx[d] - 1)
    end
    return s
end

@kernel function _mapreduce_gpu_kernel!(
        f, op, initop,
        dims::NTuple{N, Int},
        out::OT,
        inputs::IT
    ) where {N, OT <: StridedView, IT <: Tuple}

    out_linear = @index(Global, Linear)

    # Non-reduction subspace sizes (1 for reduction dims)
    nred_sizes = ntuple(Val(N)) do d
        @inbounds iszero(out.strides[d]) ? 1 : dims[d]
    end
    # Reduction subspace sizes (1 for non-reduction dims)
    red_sizes = ntuple(Val(N)) do d
        @inbounds iszero(out.strides[d]) ? dims[d] : 1
    end

    # Map out_linear → cartesian in non-reduction subspace
    nred_cidx = CartesianIndices(nred_sizes)[out_linear]
    out_parent = out.offset + 1 + _strides_dot(out.strides, nred_cidx)

    # Initialize accumulator from current output value (or apply initop)
    @inbounds acc = _gpu_init_acc(initop, out[ParentIndex(out_parent)])

    # Sequential reduction loop over reduction subspace
    @inbounds for red_linear in Base.OneTo(prod(red_sizes))
        red_cidx = CartesianIndices(red_sizes)[red_linear]
        complete_cidx = CartesianIndex(ntuple(Val(N)) do d
            @inbounds nred_cidx[d] + red_cidx[d] - 1
        end)

        val = f(ntuple(Val(length(inputs))) do m
            @inbounds begin
                a = inputs[m]
                ip = a.offset + 1 + _strides_dot(a.strides, complete_cidx)
                a[ParentIndex(ip)]
            end
        end...)

        acc = _gpu_accum(op, acc, val)
    end

    @inbounds out[ParentIndex(out_parent)] = acc
end

# GPU-compatible _mapreduce: avoids scalar indexing (first(A), out[ParentIndex(1)])
# that JLArrays/real GPUs prohibit. Uses zero(T) as a proxy to infer the output
# element type without reading from the device.
function Strided._mapreduce(
        f, op, A::StridedView{T, N, <:AnyGPUArray{T}}, nt = nothing
    ) where {T, N}
    if length(A) == 0
        b = Base.mapreduce_empty(f, op, T)
        return nt === nothing ? b : op(b, nt.init)
    end

    dims = size(A)

    if nt === nothing
        a_zero = Base.mapreduce_first(f, op, zero(T))
        out = similar(parent(A), typeof(a_zero), (1,))
        Strided._init_reduction!(out, f, op, a_zero)
    else
        out = similar(parent(A), typeof(nt.init), (1,))
        fill!(out, nt.init)
    end

    Strided._mapreducedim!(f, op, nothing, dims, (sreshape(StridedView(out), one.(dims)), A))

    return Array(out)[1]
end

function Strided._mapreduce_fuse!(
        f, op, initop,
        dims::Dims{N},
        arrays::Tuple{StridedView{TO, N, <:AnyGPUArray{TO}}, Vararg{StridedView}}
    ) where {TO, N}

    out = arrays[1]
    inputs_raw = Base.tail(arrays)
    M = length(inputs_raw)
    inputs = ntuple(i -> inputs_raw[i], Val(M))

    # Number of output elements = product of non-reduction dims
    out_total = prod(ntuple(Val(N)) do d
        @inbounds iszero(out.strides[d]) ? 1 : dims[d]
    end)

    backend = KernelAbstractions.get_backend(parent(out))
    kernel! = _mapreduce_gpu_kernel!(backend)
    kernel!(f, op, initop, dims, out, inputs; ndrange = out_total)

    return nothing
end

end
