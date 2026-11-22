# Methods based on map!
Base.copy!(dst::StridedView{<:Any, N}, src::StridedView{<:Any, N}) where {N} = map!(identity, dst, src)
Base.conj!(a::StridedView{<:Real}) = a
Base.conj!(a::StridedView) = map!(conj, a, a)
LinearAlgebra.adjoint!(dst::StridedView, src::StridedView) = copy!(dst, adjoint(src))
LinearAlgebra.transpose!(C::StridedView, A::StridedView) = copy!(C, transpose(A))
Base.permutedims!(dst::StridedView, src::StridedView, p) = copy!(dst, permutedims(src, p))
Base.fill!(A::StridedView, val) = map!(Returns(val), A, A)
function Base.fill!(
        A::StridedView{<:Union{BigFloat, Complex{BigFloat}, BigInt, Complex{BigInt}}}, val
    )
    isempty(A) && return A
    _mapreduce_order!(Returns(val), nothing, nothing, (A,))
    return A
end

# This is a wrapper function intended to allow us to
# intercept "conj" and rewrite it in cases where the
# GPU compiler seemingly isn't compiling bare conj
# correctly (https://github.com/QuantumKitHub/Strided.jl/issues/63).
# It should be removed as soon as the underlying
# compilation problem is resolved. It uses the first
# argument to dispatch so that only AMD arrays are affected.
_get_op(A::StridedView) = A.op

function Base.mapreduce(f, op, A::StridedView; dims = :, kw...)
    return Base._mapreduce_dim(f, op, values(kw), A, dims)
end

function Base._mapreduce_dim(f, op, nt::NamedTuple{(:init,)}, A::StridedView, ::Colon)
    return _mapreduce(f, op, A, nt)
end
Base._mapreduce_dim(f, op, ::NamedTuple{()}, A::StridedView, ::Colon) = _mapreduce(f, op, A)

function Base._mapreduce_dim(f, op, nt::NamedTuple{(:init,)}, A::StridedView, dims)
    return Base.mapreducedim!(f, op, Base.reducedim_initarray(A, dims, nt.init), A)
end
function Base._mapreduce_dim(f, op, ::NamedTuple{()}, A::StridedView, dims)
    return Base.mapreducedim!(f, op, Base.reducedim_init(f, op, A, dims), A)
end

function Base.map(
        f::F, a1::StridedView{<:Any, N},
        A::Vararg{StridedView{<:Any, N}}
    ) where {F, N}
    T = Base.promote_op(f, eltype(a1), eltype.(A)...)
    return map!(f, similar(a1, T), a1, A...)
end

# Note that here and below we have to specialize on the `f` and `op` arguments.
# This is because the @generated kernel always specializes, so if the callers don't
# we end up with a dynamic dispatch that has to box these variables, causing allocations
# and overhead

function Base.map!(
        f::F, b::StridedView{<:Any, N}, a1::StridedView{<:Any, N},
        A::Vararg{StridedView{<:Any, N}}
    ) where {F, N}
    dims = size(b)

    # Check dimesions
    size(a1) == dims || _throw_dimensionmismatch()
    for a in A
        size(a) == dims || _throw_dimensionmismatch()
    end

    any(isequal(0), dims) && return b # don't do anything

    _mapreduce_order!(f, nothing, nothing, (b, a1, A...))

    return b
end

function _mapreduce(f::F, op::OP, A::StridedView{T}, nt = nothing) where {F, OP, T}
    if isempty(A)
        b = Base.mapreduce_empty(f, op, T)
        return nt === nothing ? b : op(b, nt.init)
    end

    dims = size(A)
    a = Base.mapreduce_first(f, op, first(A))
    a2 = nt === nothing ? a : op(a, nt.init)
    ndims(A) == 0 && return a2 # 0-dim scalar: single-element reduction is the result
    out = similar(A, typeof(a2), (1,))
    if nt === nothing
        _init_reduction!(out, f, op, a)
    else
        out[ParentIndex(1)] = nt.init
    end
    _mapreducedim!(f, op, nothing, dims, (sreshape(out, one.(dims)), A))
    return out[ParentIndex(1)]
end

function Base.mapreducedim!(
        f::F, op::OP, b::StridedView{<:Any, N},
        a1::StridedView{<:Any, N},
        A::Vararg{StridedView{<:Any, N}}
    ) where {F, OP, N}
    outdims = size(b)
    dims = map(max, outdims, map(max, map(size, (a1, A...))...))

    # Check dimensions
    Broadcast.check_broadcast_axes(map(Base.OneTo, dims), b, a1, A...)

    return _mapreducedim!(f, op, nothing, dims, (b, a1, A...))
end

function _mapreducedim!(
        f::F, op::OP, initop::I, dims::Dims, arrays::Tuple{Vararg{StridedView}}
    ) where {F, OP, I}
    if any(isequal(0), dims)
        if length(arrays[1]) != 0 && !isnothing(initop)
            map!(initop, arrays[1], arrays[1])
        end
    else
        _mapreduce_order!(f, op, initop, promoteshape(dims, arrays...))
    end
    return arrays[1]
end

# Fuse dimensions if possible: merge dimension `i` into `i-1` when the two are
# contiguous in *every* array (`s[i] == dims[i-1] * s[i-1]`).
# Merged dimensions fold their extent into `i-1` and become size 1.
@inline function _fusedims(dims, allstrides)
    @inbounds for i in length(dims):-1:2
        merge = true
        for s in allstrides
            if s[i] != dims[i - 1] * s[i - 1]
                merge = false
                break
            end
        end
        if merge
            dims = TupleTools.setindex(dims, dims[i - 1] * dims[i], i - 1)
            dims = TupleTools.setindex(dims, 1, i)
        end
    end
    return dims
end

# Construct a view with the given layout, bypassing the default constructor to avoid stride
# normalization: zero strides (broadcast or reduction dimensions) have to survive.
@inline function _relayout(
        a::StridedView{T, <:Any, A, F}, sz::NTuple{N, Int}, st::NTuple{N, Int},
        o::Int = a.offset
    ) where {T, N, A, F}
    return StridedView{T, N, A, F}(a.parent, sz, st, o, a.op)
end

# Fusing only changes sizes, so surviving dimensions retain their importance
# order. The second permutation only moves singleton dimensions to the back.
function order_and_fuse_dims(dims::Dims, allstrides::Tuple)
    stride_orders = map(indexorder, allstrides)
    loop_order = TupleTools.sortperm(_importance(dims, stride_orders); rev = true)
    ordered_dims = TupleTools.getindices(dims, loop_order)
    ordered_strides = map(s -> TupleTools.getindices(s, loop_order), allstrides)

    fused_dims = _fusedims(ordered_dims, ordered_strides)
    active_order = TupleTools.sortperm(map(d -> Int(d > 1), fused_dims); rev = true)
    return TupleTools.getindices(fused_dims, active_order),
        map(s -> TupleTools.getindices(s, active_order), ordered_strides)
end
function order_and_fuse_dims(arrays::Tuple{Vararg{StridedView}})
    dims, allstrides = order_and_fuse_dims(size(arrays[1]), map(strides, arrays))
    return map((a, s) -> _relayout(a, dims, s), arrays, allstrides)
end

# Each stride rank gets enough bits to hold all array votes without carries.
# The output array gets two votes; each input gets one.
function _importance(dims::NTuple{N, Int}, stride_orders::NTuple{M, NTuple{N, Int}}) where {N, M}
    bits_per_rank = 8 * sizeof(Int) - leading_zeros(M + 1) # ceil(Int, log2(M + 2))
    importance = 2 .* (1 .<< (bits_per_rank .* (N .- stride_orders[1])))
    for k in 2:M
        importance = importance .+ (1 .<< (bits_per_rank .* (N .- stride_orders[k])))
    end
    return importance .* (dims .> 1)
end

# Per-dimension cost used by the blocking and thread-splitting heuristics,
# derived from the minimum (doubled) stride across arrays.
_computecosts(strides) = map(a -> ifelse(iszero(a), 1, a << 1), map(min, strides...))

# Inputs with at most this many elements skip the loop ordering and blocking, and run the
# kernel in the given loop order (after fusing contiguous dimensions): at these sizes
# everything is cache resident, and the fixed cost of planning outweighs what it saves.
const MAXNAIVELENGTH = 1 << 10

# The given loop order is only used if the destination is traversed in memory order: unit
# stride along the first dimension and increasing strides along the remaining ones.
# Zero strides (reduction dimensions) beyond the first dimension are ignored.
@inline function _isnaturalorder(dims::Dims, dststrides::Dims)
    (dims[1] > 1 && dststrides[1] == 1) || return false
    s = 1
    for i in 2:length(dims)
        (dims[i] == 1 || iszero(dststrides[i])) && continue
        dststrides[i] > s || return false
        s = dststrides[i]
    end
    return true
end

# Pipeline entry point. All arrays are expected to have the same size, with broadcast and
# reduction dimensions encoded as zero strides (see `promoteshape`).
#   order → fuse → order (reshaping the views) → block → kernel
# Fusing needs ordered entries, and the final order pass brings remaining size 1 dims to the end.
# Small inputs in a reasonable loop order only get fused, see `MAXNAIVELENGTH`.
@inline function _mapreduce_order!(
        f::F, op::OP, initop::I, arrays::Tuple{Vararg{StridedView}}
    ) where {F, OP, I}
    dims = size(arrays[1])
    len = prod(dims) # callers handle empty arrays
    # A single element needs no layout planning, regardless of the number of axes.
    len == 1 && return _mapreduce_scalar!(f, op, initop, arrays)
    # fusing preserves the natural order, so it can be checked on the unfused dimensions
    if len <= MAXNAIVELENGTH && _isnaturalorder(dims, strides(arrays[1]))
        fdims = _fusedims(dims, map(strides, arrays))
        fused = map(a -> _relayout(a, fdims, strides(a)), arrays)
        return _mapreduce_kernel!(f, op, initop, nothing, fused)
    end
    return _mapreduce_block!(f, op, initop, order_and_fuse_dims(arrays))
end

# Single-element fast path: bypass @generated kernel
function _mapreduce_scalar!(f::F, op::OP, initop::I, arrays) where {F, OP, I}
    out = arrays[1]
    iout = ParentIndex(offset(out) + 1)
    v = f(map(a -> a[ParentIndex(offset(a) + 1)], Base.tail(arrays))...)
    if op === nothing
        out[iout] = v
    else
        o = initop === nothing ? out[iout] : initop(out[iout])
        out[iout] = op(o, v)
    end
    return nothing
end

const MINTHREADLENGTH = 1 << 15 # minimal length before any kind of threading is applied
function _mapreduce_block!(f::F, op::OP, initop::I, arrays) where {F, OP, I}
    dims = size(arrays[1])
    allstrides = map(strides, arrays)
    costs = _computecosts(allstrides)
    bytestrides = map((s, stride) -> s .* stride, sizeof.(eltype.(arrays)), allstrides)
    memoryregion = totalmemoryregion(dims, bytestrides)
    # When the estimated footprint fits, the full dimensions are already optimal.
    blocks = if memoryregion <= BLOCKMEMORYSIZE
        dims
    else
        strideorders = map(indexorder, allstrides)
        _computeblocks(dims, costs, bytestrides, strideorders, BLOCKMEMORYSIZE, memoryregion)
    end

    if get_num_threads() == 1 || prod(dims) <= MINTHREADLENGTH
        _mapreduce_kernel!(f, op, initop, blocks, arrays)
    elseif op !== nothing && _length(dims, allstrides[1]) == 1 # complete reduction
        out = arrays[1]
        T = eltype(out)
        spacing = isbitstype(T) ? max(1, div(64, sizeof(T))) : 1 # to avoid false sharing
        threadedout = similar(out, spacing * get_num_threads())
        iout = ParentIndex(offset(out) + 1)
        a = out[iout]
        if initop !== nothing
            a = initop(a)
        end
        _init_reduction!(threadedout, f, op, a)

        # all strides of the output are zero, so the partial results only differ by offset
        newarrays = (_relayout(threadedout, dims, allstrides[1]), Base.tail(arrays)...)
        _mapreduce_threaded!(
            f, op, nothing, blocks, costs, newarrays, get_num_threads(), spacing, 1
        )

        for i in 1:get_num_threads()
            a = op(a, threadedout[(i - 1) * spacing + 1])
        end
        out[iout] = a
    else
        costs = costs .* .!(iszero.(allstrides[1]))
        # make cost of dimensions with zero stride in output array (reduction dimensions),
        # so that they are not divided in threading (which would lead to race conditions)

        _mapreduce_threaded!(
            f, op, initop, blocks, costs, arrays, get_num_threads(), 0, 1
        )
    end
    return nothing
end

_init_reduction!(out, f, op::Union{typeof(+), typeof(Base.add_sum)}, a) = fill!(out, zero(a))
_init_reduction!(out, f, op::Union{typeof(*), typeof(Base.mul_prod)}, a) = fill!(out, one(a))
_init_reduction!(out, f, op::typeof(min), a) = fill!(out, a)
_init_reduction!(out, f, op::typeof(max), a) = fill!(out, a)
_init_reduction!(out, f, op::typeof(&), a) = fill!(out, true)
_init_reduction!(out, f, op::typeof(|), a) = fill!(out, false)
function _init_reduction!(out, f, op, a)
    return op(a, a) == a ? fill!(out, a) :
        error("unknown reduction; incompatible with multithreading")
end

# nthreads: number of threads spacing: extra addition to offset of array 1, to account for
# reduction
function _mapreduce_threaded!(
        f::F, op::OP, initop::I, blocks, costs, arrays, nthreads, spacing, taskindex
    ) where {F, OP, I}
    dims = size(arrays[1])
    if nthreads == 1 || prod(dims) <= MINTHREADLENGTH
        return _mapreduce_kernel!(f, op, initop, blocks, _shiftoutput(arrays, spacing, taskindex))
    end
    i = _lastargmax((dims .- 1) .* costs)
    if iszero(costs[i]) || dims[i] <= min(blocks[i], 1024)
        return _mapreduce_kernel!(f, op, initop, blocks, _shiftoutput(arrays, spacing, taskindex))
    end
    di = dims[i]
    ndi = di >> 1
    nnthreads = nthreads >> 1
    arrays1 = map(a -> _relayout(a, setindex(size(a), ndi, i), strides(a)), arrays)
    t = Threads.@spawn _mapreduce_threaded!(
        f, op, initop, blocks, costs, arrays1, nnthreads, spacing, taskindex
    )
    arrays2 = map(arrays) do a
        return _relayout(
            a, setindex(size(a), di - ndi, i), strides(a), offset(a) + ndi * stride(a, i)
        )
    end
    _mapreduce_threaded!(
        f, op, initop, blocks, costs, arrays2, nthreads - nnthreads, spacing,
        taskindex + nnthreads
    )
    wait(t)
    return nothing
end

@inline function _shiftoutput(arrays, spacing, taskindex)
    iszero(spacing) && return arrays
    out = arrays[1]
    shifted = _relayout(out, size(out), strides(out), offset(out) + spacing * (taskindex - 1))
    return (shifted, Base.tail(arrays)...)
end

function _mapreduce_kernel_expr(
        f, op, initop, N::Int, M::Int; unit::Bool = false, blocked::Bool = true
    )
    @assert N > 0 "Cannot generate 0-dim kernel"

    # many variables
    blockloopvars = Array{Symbol}(undef, N)
    blockdimvars = Array{Symbol}(undef, N)
    innerloopvars = Array{Symbol}(undef, N)
    initblockdimvars = Array{Symbol}(undef, N)
    initvars = Array{Symbol}(undef, N + 1)
    for i in 1:N
        blockloopvars[i] = Symbol(:J, i)
        blockdimvars[i] = Symbol(:d, i)
        innerloopvars[i] = Symbol(:j, i)
        initblockdimvars[i] = Symbol(:d′, i)
        initvars[i] = Symbol(:init, i)
    end
    initvars[N + 1] = Symbol(:init, N + 1)

    Ivars = Array{Symbol}(undef, M)
    Avars = Array{Symbol}(undef, M)
    stridevars = Array{Symbol}(undef, (N, M))
    for j in 1:M
        Ivars[j] = Symbol(:I, j)
        Avars[j] = Symbol(:A, j)
        for i in 1:N
            stridevars[i, j] = Symbol(:stride_, i, :_, j)
        end
    end

    # useful expressions
    pre1ex = Expr(:block)
    pre2ex = Expr(:block)
    pre3ex = Expr(:block)
    push!(pre1ex.args, :(dims = size(arrays[1])))
    for j in 1:M
        push!(pre1ex.args, :($(Avars[j]) = arrays[$j]))
        push!(pre3ex.args, :($(Ivars[j]) = offset($(Avars[j])) + 1))
        for i in 1:N
            push!(pre2ex.args, :($(stridevars[i, j]) = stride($(Avars[j]), $i)))
        end
    end

    fcallex = Expr(:call, :f)
    for j in 2:M
        push!(fcallex.args, :($(Avars[j])[ParentIndex($(Ivars[j]))]))
    end
    lhsex = :(A1[ParentIndex($(Ivars[1]))])

    stepstride1ex = Vector{Expr}(undef, N)
    stepstride2ex = Vector{Expr}(undef, N)
    returnstride1ex = Vector{Expr}(undef, N)
    returnstride2ex = Vector{Expr}(undef, N)
    for i in 1:N
        stepstride1ex[i] = :($(Ivars[1]) += $(stridevars[i, 1]))
        returnstride1ex[i] = :($(Ivars[1]) -= $(blockdimvars[i]) * $(stridevars[i, 1]))
        stepex = Expr(:block)
        returnex = Expr(:block)
        for j in 2:M
            push!(stepex.args, :($(Ivars[j]) += $(stridevars[i, j])))
            push!(returnex.args, :($(Ivars[j]) -= $(blockdimvars[i]) * $(stridevars[i, j])))
        end
        stepstride2ex[i] = stepex
        returnstride2ex[i] = returnex
    end

    outerstepstrideex = Vector{Expr}(undef, N)
    outerreturnstrideex = Vector{Expr}(undef, N)
    for i in 1:N
        stepex = Expr(:block)
        returnex = Expr(:block)
        for j in 1:M
            push!(stepex.args, :($(Ivars[j]) += $(blockdimvars[i]) * $(stridevars[i, j])))
            push!(returnex.args, :($(Ivars[j]) -= dims[$i] * $(stridevars[i, j])))
        end
        outerstepstrideex[i] = stepex
        outerreturnstrideex[i] = returnex
    end

    # Unit-stride fast path for the innermost (vectorized) loop dimension.
    # We special-case for contiguous loads/stores to avoid SIMD gather/scatter
    # in favor of SIMD load/store, which streams memory more efficiently.
    unitstep1ex = :($(Ivars[1]) += 1)
    unitstep2ex = Expr(:block)
    for j in 2:M
        push!(unitstep2ex.args, :($(Ivars[j]) += 1))
    end
    firststrides = Expr(:tuple, (stridevars[1, j] for j in 1:M)...)
    unitstridecond = :(all(==(1), $firststrides))

    i = 1
    if op == Nothing
        # Non-reducing (map!/permute/...): each destination element is written once,
        # so skip the reduction-only `iszero` hoist below. Whether the innermost loop
        # has unit stride is loop invariant, and is selected by the caller via `unit`.
        ex = Expr(:(=), lhsex, fcallex)
        if N >= 1
            step1ex, step2ex = unit ? (unitstep1ex, unitstep2ex) : (stepstride1ex[i], stepstride2ex[i])
            ex = quote
                @simd for $(innerloopvars[i]) in Base.OneTo($(blockdimvars[i]))
                    $ex
                    $step1ex
                    $step2ex
                end
                $(returnstride1ex[i])
                $(returnstride2ex[i])
            end
        end
    else
        ex = Expr(:(=), lhsex, Expr(:call, :op, lhsex, fcallex))
        exa = Expr(:(=), :a, Expr(:call, :op, :a, fcallex))
        if N >= 1
            ex = quote
                if iszero($(stridevars[1, 1])) # explicitly hoist A1[I1] out of loop
                    a = $lhsex
                    @simd for $(innerloopvars[i]) in Base.OneTo($(blockdimvars[i]))
                        $exa
                        $(stepstride2ex[i])
                    end
                    $lhsex = a
                    $(returnstride2ex[i])
                elseif $unitstridecond
                    @simd for $(innerloopvars[i]) in Base.OneTo($(blockdimvars[i]))
                        $ex
                        $unitstep1ex
                        $unitstep2ex
                    end
                    $(returnstride1ex[i])
                    $(returnstride2ex[i])
                else
                    @simd for $(innerloopvars[i]) in Base.OneTo($(blockdimvars[i]))
                        $ex
                        $(stepstride1ex[i])
                        $(stepstride2ex[i])
                    end
                    $(returnstride1ex[i])
                    $(returnstride2ex[i])
                end
            end
        end
    end
    for outer i in 2:N
        ex = quote
            for $(innerloopvars[i]) in Base.OneTo($(blockdimvars[i]))
                $ex
                $(stepstride1ex[i])
                $(stepstride2ex[i])
            end
            $(returnstride1ex[i])
            $(returnstride2ex[i])
        end
    end

    if initop !== Nothing
        initex = Expr(:(=), lhsex, Expr(:call, :initop, lhsex))
        i = 1
        if N >= 1
            initex = quote
                $(initblockdimvars[i]) = ifelse(
                    iszero($(stridevars[i, 1])), 1,
                    $(blockdimvars[i])
                )
                @simd for $(innerloopvars[i]) in Base.OneTo($(initblockdimvars[i]))
                    $initex
                    $(stepstride1ex[i])
                end
                $(Ivars[1]) -= $(initblockdimvars[i]) * $(stridevars[i, 1])
            end
        end
        for outer i in 2:N
            initex = quote
                $(initblockdimvars[i]) = ifelse(
                    iszero($(stridevars[i, 1])), 1,
                    $(blockdimvars[i])
                )
                for $(innerloopvars[i]) in Base.OneTo($(initblockdimvars[i]))
                    $initex
                    $(stepstride1ex[i])
                end
                $(Ivars[1]) -= $(initblockdimvars[i]) * $(stridevars[i, 1])
            end
        end
        ex = quote
            if $(initvars[1])
                $initex
            end
            $ex
        end
    end

    if !blocked
        # a single block spanning the full dimensions: no block loops needed
        dimsex = Expr(:block, (:($(blockdimvars[i]) = dims[$i]) for i in 1:N)...)
        return quote
            $pre1ex
            $pre2ex
            $pre3ex
            $dimsex
            $(initvars[1]) = true
            @inbounds $ex
            return nothing
        end
    end

    if initop === Nothing
        for outer i in 1:N
            ex = quote
                $(blockloopvars[i]) = 1
                while $(blockloopvars[i]) <= dims[$i]
                    $(blockdimvars[i]) = min(blocks[$i], dims[$i] - $(blockloopvars[i]) + 1)
                    $ex
                    $(outerstepstrideex[i])
                    $(blockloopvars[i]) += blocks[$i]
                end
                $(outerreturnstrideex[i])
            end
        end
        ex = quote
            $pre1ex
            $pre2ex
            $pre3ex
            @inbounds $ex
            return nothing
        end
    else
        for outer i in 1:N
            ex = quote
                $(initvars[i]) = $(initvars[i + 1])
                $(blockloopvars[i]) = 1
                while $(blockloopvars[i]) <= dims[$i]
                    $(blockdimvars[i]) = min(blocks[$i], dims[$i] - $(blockloopvars[i]) + 1)
                    $ex
                    $(initvars[i]) &= $(stridevars[i, 1]) > 0
                    $(outerstepstrideex[i])
                    $(blockloopvars[i]) += blocks[$i]
                end
                $(outerreturnstrideex[i])
            end
        end
        ex = quote
            $pre1ex
            $pre2ex
            $pre3ex
            $(initvars[N + 1]) = true
            @inbounds $ex
            return nothing
        end
    end
    return ex
end

# The loop bounds, strides and offsets are all taken from `arrays`, which have to share the
# same size; `blocks` holds the block size for each dimension, or is `nothing` to loop over
# the full dimensions without blocking.
@generated function _mapreduce_kernel!(
        (f), (op), (initop), blocks::Union{Nothing, Dims},
        arrays::Tuple{StridedView, Vararg{StridedView}}
    )
    N = ndims(fieldtype(arrays, 1))
    M = fieldcount(arrays)
    blocked = blocks !== Nothing
    op === Nothing || return _mapreduce_kernel_expr(f, op, initop, N, M; blocked)
    # select the unit-stride variant of the innermost loop once, instead of in every iteration
    unitcond = Expr(:call, :all, :(==(1)), Expr(:tuple, (:(stride(arrays[$j], 1)) for j in 1:M)...))
    return quote
        $(blocked ? nothing : Expr(:meta, :inline))
        if $unitcond
            $(_mapreduce_kernel_expr(f, op, initop, N, M; unit = true, blocked))
        else
            $(_mapreduce_kernel_expr(f, op, initop, N, M; unit = false, blocked))
        end
    end
end

function indexorder(strides::NTuple{N, Int}) where {N}
    # Rank absolute nonzero strides. Ties share a rank; zero strides get rank 1.
    return ntuple(Val(N)) do i
        si = abs(strides[i])
        iszero(si) && return 1
        k = 1
        for s in strides
            if !iszero(s) && abs(s) < si
                k += 1
            end
        end
        return k
    end
end

function _length(dims::Tuple, strides::Tuple)
    return ifelse(iszero(strides[1]), 1, dims[1]) *
        _length(Base.tail(dims), Base.tail(strides))
end
_length(dims::Tuple{}, strides::Tuple{}) = 1
function _maxlength(dims::Tuple, strides::Tuple{Vararg{Tuple}})
    return maximum(broadcast(_length, (dims,), strides))
end

function _lastargmax(t::Tuple)
    i = 1
    for j in 2:length(t)
        @inbounds if t[j] >= t[i]
            i = j
        end
    end
    return i
end

const BLOCKMEMORYSIZE = 1 << 15 # L1 cache size in bytes
function _computeblocks(
        dims::Tuple{}, costs::Tuple{},
        bytestrides::Tuple{Vararg{Tuple{}}},
        strideorders::Tuple{Vararg{Tuple{}}},
        blocksize::Int = BLOCKMEMORYSIZE
    )
    return ()
end

function _computeblocks(
        dims::NTuple{N, Int}, costs::NTuple{N, Int},
        bytestrides::Tuple{Vararg{NTuple{N, Int}}},
        strideorders::Tuple{Vararg{NTuple{N, Int}}},
        blocksize::Int = BLOCKMEMORYSIZE,
        memoryregion::Int = totalmemoryregion(dims, bytestrides)
    ) where {N}
    if memoryregion <= blocksize
        return dims
    end
    minstrideorder = minimum(minimum.(strideorders))
    if all(isequal(minstrideorder), first.(strideorders))
        d1 = dims[1]
        dr = _computeblocks(
            tail(dims), tail(costs),
            map(tail, bytestrides), map(tail, strideorders), blocksize
        )
        return (d1, dr...)
    end

    if minimum(minimum.(bytestrides)) > blocksize
        return ntuple(n -> 1, Val(N))
    end

    # reduce dims to find appropriate blocks
    blocks = dims
    while totalmemoryregion(blocks, bytestrides) >= 2 * blocksize
        i = _lastargmax((blocks .- 1) .* costs)
        blocks = TupleTools.setindex(blocks, (blocks[i] + 1) >> 1, i)
    end
    while totalmemoryregion(blocks, bytestrides) > blocksize
        i = _lastargmax((blocks .- 1) .* costs)
        blocks = TupleTools.setindex(blocks, blocks[i] - 1, i)
    end
    return blocks
end

const _cachelinelength = 64
function totalmemoryregion(dims, bytestrides)
    memoryregion = 0
    for i in 1:length(bytestrides)
        strides = bytestrides[i]
        numcontigeouscachelines = 0
        numcachelineblocks = 1
        for (d, s) in zip(dims, strides)
            if s < _cachelinelength
                numcontigeouscachelines += (d - 1) * s
            else
                numcachelineblocks *= d
            end
        end
        numcontigeouscachelines = div(numcontigeouscachelines, _cachelinelength) + 1
        memoryregion += _cachelinelength * numcontigeouscachelines * numcachelineblocks
    end
    return memoryregion
end
