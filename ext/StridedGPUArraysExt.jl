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

@inline function cartesian2parent(strides::NTuple{N, Int}, cidx::CartesianIndex{N}) where {N}
    s = 0
    for d in Base.OneTo(N)
        @inbounds s += strides[d] * (cidx[d] - 1)
    end
    return s
end

@kernel function _mapreduce_gpu_kernel!(
        f, op, initop,
        dims_red, strides, offsets,
        arrays
    )

    I_out = @index(Global, Cartesian)

    # Compute parent index for current index.
    Is_parent = cartesian2parent.(strides, Ref(I_out)) .+ offsets .+ 1

    # Initialize accumulator from current output value (or apply initop)
    out = arrays[1]
    out_I_parent = Is_parent[1]
    @inbounds acc = _gpu_init_acc(initop, out[ParentIndex(out_I_parent)])

    inputs = Base.tail(arrays)
    inputs_I_parent = Base.tail(Is_parent)
    inputs_strides = Base.tail(strides)

    for I_red in CartesianIndices(dims_red)
        # Compute parent index for current reduction index
        Is_red_parent = cartesian2parent.(inputs_strides, Ref(I_red))
        # Get values from each input array, apply map function, and accumulate
        vals = getindex.(inputs, ParentIndex.(inputs_I_parent .+ Is_red_parent))
        acc = _gpu_accum(op, acc, f(vals...))
    end
    # Write back result to output array
    @inbounds out[ParentIndex(out_I_parent)] = acc
end

# GPU-compatible _mapreduce: avoids scalar indexing (first(A), out[ParentIndex(1)])
# that JLArrays/real GPUs prohibit. Mirrors GPUArrays' neutral_element approach:
# infer output type via Broadcast machinery, look up the neutral element (errors on
# unknown ops), fill the output buffer, then read back a single scalar via Array().
function Strided._mapreduce(
        f, op, A::GPUStridedView{T, N}, nt = nothing
    ) where {T, N}
    if isempty(A)
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

function Strided._mapreduce_block!(
        f, op, initop,
        dims::Dims{N},
        strides, offsets, costs,
        arrays::Tuple{GPUStridedView{TO, N}, Vararg{GPUStridedView{<:Any, N}}}
    ) where {TO, N}

    out = arrays[1]
    out_strides = strides[1]

    # Number of output elements = product of non-reduction dims
    dims_out = ntuple(Val(N)) do d
        @inbounds iszero(out_strides[d]) ? 1 : dims[d]
    end
    dims_red = ntuple(Val(N)) do d
        @inbounds iszero(out_strides[d]) ? dims[d] : 1
    end

    backend = KernelAbstractions.get_backend(parent(out))
    kernel! = _mapreduce_gpu_kernel!(backend)
    kernel!(f, op, initop, dims_red, strides, offsets, arrays; ndrange = dims_out)

    return nothing
end

end
