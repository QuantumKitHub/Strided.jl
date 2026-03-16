module StridedGPUArraysExt

using Strided, GPUArrays, LinearAlgebra
using GPUArrays: Adapt, KernelAbstractions
using GPUArrays.KernelAbstractions: @kernel, @index
using StridedViews: ParentIndex

ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

# StridedView backed by any GPU array type, with element type linked to the parent.
const GPUStridedView{T, N} = StridedView{T, N, <:AnyGPUArray{T}}

KernelAbstractions.get_backend(sv::GPUStridedView) = KernelAbstractions.get_backend(parent(sv))

# Conversion to CPU Array: materialise into a contiguous GPU array first (so the
# GPU-to-GPU copy! path is used), then let the GPU array type handle the transfer.
function Base.Array(a::GPUStridedView)
    b = similar(parent(a), eltype(a), size(a))
    copy!(StridedView(b), a)
    return Array(b)
end

function Strided.__mul!(
        C::GPUStridedView{TC, 2},
        A::GPUStridedView{TA, 2},
        B::GPUStridedView{TB, 2},
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
        complete_cidx = CartesianIndex(
            ntuple(Val(N)) do d
                @inbounds nred_cidx[d] + red_cidx[d] - 1
            end
        )

        val = f(
            ntuple(Val(length(inputs))) do m
                @inbounds begin
                    a = inputs[m]
                    ip = a.offset + 1 + _strides_dot(a.strides, complete_cidx)
                    a[ParentIndex(ip)]
                end
            end...
        )

        acc = _gpu_accum(op, acc, val)
    end

    @inbounds out[ParentIndex(out_parent)] = acc
end

# GPU-compatible _mapreduce: avoids scalar indexing (first(A), out[ParentIndex(1)])
# that JLArrays/real GPUs prohibit. Mirrors GPUArrays' neutral_element approach:
# infer output type via Broadcast machinery, look up the neutral element (errors on
# unknown ops), fill the output buffer, then read back a single scalar via Array().
function Strided._mapreduce(
        f, op, A::GPUStridedView{T, N}, nt = nothing
    ) where {T, N}
    if length(A) == 0
        b = Base.mapreduce_empty(f, op, T)
        return nt === nothing ? b : op(b, nt.init)
    end

    dims = size(A)

    if nt === nothing
        ET = Base.Broadcast.combine_eltypes(f, (A,))
        ET = Base.promote_op(op, ET, ET)
        (ET === Union{} || ET === Any) &&
            error("cannot infer output element type for mapreduce; pass an explicit `init`")
        init = GPUArrays.neutral_element(op, ET)
    else
        ET = typeof(nt.init)
        init = nt.init
    end

    out = similar(parent(A), ET, (1,))
    fill!(out, init)

    Strided._mapreducedim!(f, op, nothing, dims, (sreshape(StridedView(out), one.(dims)), A))

    return Array(out)[1]
end

function Strided._mapreduce_fuse!(
        f, op, initop,
        dims::Dims{N},
        arrays::Tuple{GPUStridedView{TO, N}, Vararg{GPUStridedView{<:Any, N}}}
    ) where {TO, N}

    out = arrays[1]
    inputs_raw = Base.tail(arrays)
    M = length(inputs_raw)
    inputs = ntuple(i -> inputs_raw[i], Val(M))

    # Number of output elements = product of non-reduction dims
    out_total = prod(
        ntuple(Val(N)) do d
            @inbounds iszero(out.strides[d]) ? 1 : dims[d]
        end
    )

    backend = KernelAbstractions.get_backend(parent(out))
    kernel! = _mapreduce_gpu_kernel!(backend)
    kernel!(f, op, initop, dims, out, inputs; ndrange = out_total)

    return nothing
end

end
