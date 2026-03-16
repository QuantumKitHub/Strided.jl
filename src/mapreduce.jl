# Methods based on map!
Base.copy!(dst::StridedView{<:Any, N}, src::StridedView{<:Any, N}) where {N} = map!(identity, dst, src)
Base.conj!(a::StridedView{<:Real}) = a
Base.conj!(a::StridedView) = map!(conj, a, a)
LinearAlgebra.adjoint!(dst::StridedView, src::StridedView) = copy!(dst, adjoint(src))
LinearAlgebra.transpose!(C::StridedView, A::StridedView) = copy!(C, transpose(A))
Base.permutedims!(dst::StridedView, src::StridedView, p) = copy!(dst, permutedims(src, p))
Base.fill!(A::StridedView, val) = map!(Returns(val), A)

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
    T = Base.promote_eltype(a1, A...)
    return map!(f, similar(a1, T), a1, A...)
end

function Base.map!(
        f::F, b::StridedView{<:Any, N}, a1::StridedView{<:Any, N},
        A::Vararg{StridedView{<:Any, N}}
    ) where {F, N}
    dims = size(b)

    # Check dimesions
    size(a1) == dims || throw(DimensionMismatch())
    for a in A
        size(a) == dims || throw(DimensionMismatch())
    end

    any(isequal(0), dims) && return b # don't do anything

    _mapreduce_fuse!(f, nothing, nothing, dims, (b, a1, A...))

    return b
end

function _mapreduce(f, op, A::StridedView{T}, nt = nothing) where {T}
    if length(A) == 0
        b = Base.mapreduce_empty(f, op, T)
        return nt === nothing ? b : op(b, nt.init)
    end

    dims = size(A)
    a = Base.mapreduce_first(f, op, first(A))
    a2 = nt === nothing ? a : op(a, nt.init)
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
        f, op, b::StridedView{<:Any, N},
        a1::StridedView{<:Any, N},
        A::Vararg{StridedView{<:Any, N}}
    ) where {N}
    outdims = size(b)
    dims = map(max, outdims, map(max, map(size, (a1, A...))...))

    # Check dimensions
    Broadcast.check_broadcast_axes(map(Base.OneTo, dims), b, a1, A...)

    return _mapreducedim!(f, op, nothing, dims, (b, a1, A...))
end

function _mapreducedim!(
        (f), (op), (initop),
        dims::Dims, arrays::Tuple{Vararg{StridedView}}
    )
    if any(isequal(0), dims)
        if length(arrays[1]) != 0 && !isnothing(initop)
            map!(initop, arrays[1], arrays[1])
        end
    else
        _mapreduce_fuse!(f, op, initop, dims, promoteshape(dims, arrays...))
    end
    return arrays[1]
end

function _mapreduce_fuse!(
        (f), (op), (initop),
        dims::Dims, arrays::Tuple{Vararg{StridedView}}
    )
    # Fuse dimensions if possible: assume that at least one array, e.g. the output array in
    # arrays[1], has its strides sorted
    allstrides = map(strides, arrays)
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
    return _mapreduce_order!(f, op, initop, dims, allstrides, arrays)
end

function _mapreduce_order!(
        (f), (op), (initop),
        dims, strides, arrays
    )
    M = length(arrays)
    N = length(dims)
    # sort order of loops/dimensions by modelling the importance of each dimension
    g = 8 * sizeof(Int) - leading_zeros(M + 1) # ceil(Int, log2(M+2)) # to account for the fact
    # that there are M arrays, where the first one is counted with a factor 2
    importance = 2 .* (1 .<< (g .* (N .- indexorder(strides[1]))))  # first array is output
    # and is more important by a factor 2
    for k in 2:M
        importance = importance .+ (1 .<< (g .* (N .- indexorder(strides[k]))))
    end

    importance = importance .* (dims .> 1) # put dims 1 at the back
    p = TupleTools.sortperm(importance; rev = true)
    dims = TupleTools.getindices(dims, p)
    strides = broadcast(TupleTools.getindices, strides, (p,))
    offsets = map(offset, arrays)
    costs = map(a -> ifelse(iszero(a), 1, a << 1), map(min, strides...))
    return _mapreduce_block!(f, op, initop, dims, strides, offsets, costs, arrays)
end

const MINTHREADLENGTH = 1 << 15 # minimal length before any kind of threading is applied
function _mapreduce_block!(
        (f), (op), (initop),
        dims, strides, offsets, costs, arrays
    )
    bytestrides = map((s, stride) -> s .* stride, sizeof.(eltype.(arrays)), strides)
    strideorders = map(indexorder, strides)
    blocks = _computeblocks(dims, costs, bytestrides, strideorders)

    # t = @elapsed _computeblocks(dims, costs, bytestrides, strideorders)
    # println("_computeblocks time: $t")

    if get_num_threads() == 1 || prod(dims) <= MINTHREADLENGTH
        _mapreduce_kernel!(f, op, initop, dims, blocks, arrays, strides, offsets)
    elseif op !== nothing && _length(dims, strides[1]) == 1 # complete reduction
        T = eltype(arrays[1])
        spacing = isbitstype(T) ? max(1, div(64, sizeof(T))) : 1 # to avoid false sharing
        threadedout = similar(arrays[1], spacing * get_num_threads())
        a = arrays[1][ParentIndex(1)]
        if initop !== nothing
            a = initop(a)
        end
        _init_reduction!(threadedout, f, op, a)

        newarrays = (threadedout, Base.tail(arrays)...)
        _mapreduce_threaded!(
            f, op, nothing, dims, blocks, strides, offsets, costs,
            newarrays, get_num_threads(), spacing, 1
        )

        for i in 1:get_num_threads()
            a = op(a, threadedout[(i - 1) * spacing + 1])
        end
        arrays[1][ParentIndex(1)] = a
    else
        costs = costs .* .!(iszero.(strides[1]))
        # make cost of dimensions with zero stride in output array (reduction dimensions),
        # so that they are not divided in threading (which would lead to race conditions)

        _mapreduce_threaded!(
            f, op, initop, dims, blocks, strides, offsets, costs, arrays,
            get_num_threads(), 0, 1
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
        (f), (op), (initop),
        dims, blocks, strides, offsets, costs, arrays, nthreads,
        spacing, taskindex
    )
    if nthreads == 1 || prod(dims) <= MINTHREADLENGTH
        offset1 = offsets[1] + spacing * (taskindex - 1)
        spacedoffsets = (offset1, Base.tail(offsets)...)
        _mapreduce_kernel!(f, op, initop, dims, blocks, arrays, strides, spacedoffsets)
    else
        i = _lastargmax((dims .- 1) .* costs)
        if costs[i] == 0 || dims[i] <= min(blocks[i], 1024)
            offset1 = offsets[1] + spacing * (taskindex - 1)
            spacedoffsets = (offset1, Base.tail(offsets)...)
            _mapreduce_kernel!(f, op, initop, dims, blocks, arrays, strides, spacedoffsets)
        else
            di = dims[i]
            ndi = di >> 1
            nnthreads = nthreads >> 1
            newdims = setindex(dims, ndi, i)
            newoffsets = offsets
            t = Threads.@spawn _mapreduce_threaded!(
                f, op, initop, newdims, blocks, strides,
                newoffsets, costs, arrays, nnthreads,
                spacing, taskindex
            )
            stridesi = getindex.(strides, i)
            newoffsets2 = offsets .+ ndi .* stridesi
            newdims2 = setindex(dims, di - ndi, i)
            nnthreads2 = nthreads - nnthreads
            _mapreduce_threaded!(
                f, op, initop, newdims2, blocks, strides, newoffsets2,
                costs, arrays, nnthreads2, spacing, taskindex + nnthreads
            )
            wait(t)
        end
    end
    return nothing
end

@generated function _mapreduce_kernel!(
        (f), (op),
        (initop), dims::NTuple{N, Int},
        blocks::NTuple{N, Int},
        arrays::NTuple{M, StridedView},
        strides::NTuple{M, NTuple{N, Int}},
        offsets::NTuple{M, Int}
    ) where {N, M}

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
    for j in 1:M
        push!(pre1ex.args, :($(Avars[j]) = arrays[$j]))
        push!(pre3ex.args, :($(Ivars[j]) = offsets[$j] + 1))
        for i in 1:N
            push!(pre2ex.args, :($(stridevars[i, j]) = strides[$j][$i]))
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

    if op == Nothing
        ex = Expr(:(=), lhsex, fcallex)
        exa = Expr(:(=), :a, fcallex)
    else
        ex = Expr(:(=), lhsex, Expr(:call, :op, lhsex, fcallex))
        exa = Expr(:(=), :a, Expr(:call, :op, :a, fcallex))
    end
    i = 1
    if N >= 1
        ex = quote
            if $(stridevars[1, 1]) == 0 # explicitly hoist A1[I1] out of loop
                a = $lhsex
                @simd for $(innerloopvars[i]) in Base.OneTo($(blockdimvars[i]))
                    $exa
                    $(stepstride2ex[i])
                end
                $lhsex = a
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
                    $(stridevars[i, 1]) == 0, 1,
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
                    $(stridevars[i, 1]) == 0, 1,
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

    if initop === Nothing
        for outer i in 1:N
            ex = quote
                for $(blockloopvars[i]) in 1:blocks[$i]:dims[$i]
                    $(blockdimvars[i]) = min(blocks[$i], dims[$i] - $(blockloopvars[i]) + 1)
                    $ex
                    $(outerstepstrideex[i])
                end
                $(outerreturnstrideex[i])
            end
        end
        ex = quote
            $pre1ex
            $pre2ex
            $pre3ex
            @inbounds $ex
            return A1
        end
    else
        for outer i in 1:N
            ex = quote
                $(initvars[i]) = $(initvars[i + 1])
                for $(blockloopvars[i]) in 1:blocks[$i]:dims[$i]
                    $(blockdimvars[i]) = min(blocks[$i], dims[$i] - $(blockloopvars[i]) + 1)
                    $ex
                    $(initvars[i]) &= $(stridevars[i, 1]) > 0
                    $(outerstepstrideex[i])
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
            return A1
        end
    end
    return ex
end

function indexorder(strides::NTuple{N, Int}) where {N}
    # returns order such that strides[i] is the order[i]th smallest element of strides, not
    # counting zero strides zero strides have order 1
    return ntuple(Val(N)) do i
        si = abs(strides[i])
        si == 0 && return 1
        k = 1
        for s in strides
            if s != 0 && abs(s) < si
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
        blocksize::Int = BLOCKMEMORYSIZE
    ) where {N}
    if totalmemoryregion(dims, bytestrides) <= blocksize
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
