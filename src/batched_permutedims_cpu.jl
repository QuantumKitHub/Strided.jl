# CPU executor: `_batched_permute!` for plain `StridedView`s (`plan.deviceid === nothing`).
# Global tile ids `0:totaltiles-1` are consecutive per tensor (tensor b owns
# `prefix[b]:prefix[b+1]-1`); each is located, decoded and copied as one tile.

# Decode local tile id `l` (mixed radix over `tilecounts`, axis 1 fastest) into stride-multiplied
# source/destination offsets and the tile's extents (`H[g]`, or the remainder at the tensor's
# edge). Recursion over `Base.tail` unrolls fully for a static `N`: one specialization per
# reduced rank, none per shape or batch.
@inline function _tile_geom(
        dims::NTuple{N, Int}, tilecounts::NTuple{N, Int}, srcstrides::NTuple{N, Int},
        dststrides::NTuple{N, Int}, H::NTuple{N, Int}, l::Int
    ) where {N}
    tc = tilecounts[1]
    o = (l % tc) * H[1]
    r = l ÷ tc
    d1 = min(H[1], dims[1] - o)
    soffrest, doffrest, drest = _tile_geom(
        Base.tail(dims), Base.tail(tilecounts), Base.tail(srcstrides),
        Base.tail(dststrides), Base.tail(H), r
    )
    return o * srcstrides[1] + soffrest, o * dststrides[1] + doffrest, (d1, drest...)
end
@inline _tile_geom(::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, ::Tuple{}, l::Int) = (0, 0, ())

# Absolute source/destination base offsets and extents of tile `l` of one tensor. Shared by the
# CPU runner and both GPU kernels, so every executor agrees on tile geometry by construction.
@inline function _tile_offsets(desc::BatchedPermuteDesc{N}, H::NTuple{N, Int}, l::Int) where {N}
    dsoff, ddoff, d = _tile_geom(desc.dims, desc.tilecounts, desc.srcstrides, desc.dststrides, H, l)
    return desc.srcoffset + dsoff, desc.dstoffset + ddoff, d
end

# One tile through Strided's serial tiled kernel (no reduction, so each destination element is
# written once). Unscaled: `f = identity` is exactly a strided copy. Scaled with `beta == 0`:
# `a * x`, still never reading `dst`. Otherwise `dst` is also the third input, read and
# rewritten at the same index in one call, as Strided's own `axpby!` broadcast does.
@inline function _batched_tile_kernel!(
        ::Nothing, ::Int, d, blocks, dst::StridedView, src::StridedView, dstr, sstr, doff::Int, soff::Int
    )
    _mapreduce_kernel!(identity, nothing, nothing, d, blocks, (dst, src), (dstr, sstr), (doff, soff))
    return nothing
end
@inline function _batched_tile_kernel!(
        coeffs::Tuple, b::Int, d, blocks, dst::StridedView, src::StridedView, dstr, sstr, doff::Int, soff::Int
    )
    a = coeffs[1][b]
    bb = coeffs[2][b]
    if iszero(bb)
        _mapreduce_kernel!(Base.Fix1(*, a), nothing, nothing, d, blocks, (dst, src), (dstr, sstr), (doff, soff))
    else
        _mapreduce_kernel!(
            (x, y) -> _batched_axpby(a, x, bb, y), nothing, nothing, d, blocks,
            (dst, src, dst), (dstr, sstr, dstr), (doff, soff, doff)
        )
    end
    return nothing
end

@inline function _batched_run_tile!(
        dst::Vector{<:StridedView}, src::Vector{<:StridedView},
        plan::BatchedPermutePlan, b::Int, l::Int, coeffs
    )
    desc = plan.descs[b]
    soff, doff, d = _tile_offsets(desc, plan.tile, l)
    _batched_tile_kernel!(
        coeffs, b, d, plan.cpublocks[b], dst[b], src[b], desc.dststrides, desc.srcstrides, doff, soff
    )
    return nothing
end

# Run global tile ids `[ustart, uend)`, which may span tensors: locate the first tensor once,
# then advance `b` across prefix boundaries instead of re-searching per tile.
function _batched_chunk!(
        dst::Vector{<:StridedView}, src::Vector{<:StridedView},
        plan::BatchedPermutePlan, ustart::Int, uend::Int, coeffs
    )
    ustart >= uend && return nothing
    prefix = plan.prefix
    b = searchsortedlast(prefix, ustart)
    for u in ustart:(uend - 1)
        while u >= prefix[b + 1]
            b += 1
        end
        _batched_run_tile!(dst, src, plan, b, u - prefix[b], coeffs)
    end
    return nothing
end

# Worker boundaries balancing *elements*, not tile counts: tiles differ in size across tensors
# and are contiguous per tensor, so equal id ranges can hand nearly all the work to one worker.
# A tensor straddling a worker's element budget is split proportionally (assuming roughly
# equal tiles, which holds except at its edge; this only picks boundaries, never addresses).
function _batched_worker_bounds(plan::BatchedPermutePlan, nw::Int)
    B = length(plan.descs)
    totaltiles = plan.totaltiles
    total = plan.totalelements
    bounds = Vector{Int}(undef, nw + 1)
    bounds[1] = 0
    bounds[end] = totaltiles
    b = 1
    cum = 0 # elements in tensors fully before tensor b
    for i in 1:(nw - 1)
        target = (total * i) ÷ nw
        while true
            elemsb = _cprod(plan.descs[b].dims)
            Tb = plan.prefix[b + 1] - plan.prefix[b]
            if b < B && cum + elemsb <= target
                cum += elemsb
                b += 1
                continue
            end
            pertile = Tb == 0 ? 0.0 : elemsb / Tb
            localtiles = pertile <= 0 ? 0 : round(Int, (target - cum) / pertile)
            bounds[i + 1] = plan.prefix[b] + clamp(localtiles, 0, Tb)
            break
        end
    end
    for i in 2:nw # rounding could produce a tiny local decrease; disallow it
        bounds[i] = clamp(bounds[i], bounds[i - 1], totaltiles)
    end
    return bounds
end

# Below `MINTHREADLENGTH` or with one worker, run every tile on the calling task; otherwise split
# into `min(nthreads, totaltiles)` element-balanced ranges, one task each (the caller runs the
# first). No per-worker scratch and no state keyed by `threadid()`, so task migration is harmless.
function _batched_permute!(
        plan::BatchedPermutePlan, dst::Vector{<:StridedView}, src::Vector{<:StridedView}, coeffs
    )
    totaltiles = plan.totaltiles
    nw = get_num_threads()
    if nw == 1 || plan.totalelements <= MINTHREADLENGTH
        _batched_chunk!(dst, src, plan, 0, totaltiles, coeffs)
    else
        nw = min(nw, totaltiles)
        bounds = _batched_worker_bounds(plan, nw)
        @sync begin
            for i in 2:nw
                Threads.@spawn _batched_chunk!(dst, src, plan, bounds[i], bounds[i + 1], coeffs)
            end
            _batched_chunk!(dst, src, plan, bounds[1], bounds[2], coeffs)
        end
    end
    return dst
end
