# Batched out-of-place permutation: `dsts[b] = permutedims(srcs[b], perm)` for every `b`,
# with one shared `perm`, one common rank and element type, and free per-tensor sizes. This
# file only plans; the executors are `batched_permutedims_cpu.jl` (CPU) and
# `ext/StridedGPUArraysExt_batched.jl` (GPU), both reached via `_batched_permute!(plan, dst, src)`.
#
# Addressing uses each view's own `strides(...)` as-is: no density check and NO ALIASING
# CHECK. Overlapping destinations, or a destination overlapping a source, is undefined
# behavior (a silently wrong result, not an error). Only source/source aliasing is safe.

@enum BatchedPermuteFamily BP_COPY BP_PAYLOAD BP_TRANSPOSE

# `BP_AUTO` lets the backend decide. The other values are GPU-only: on a CPU batch they are an
# error, never a silent fallback (a silent no-op would make strategy comparisons misleading).
@enum BatchedPermuteStrategy BP_AUTO BP_ELEMENTWISE BP_THREADTILE BP_GROUPTILE

# Per-tensor addressing data in the batch's shared reduced-axis order: coordinate `c` lives at
# `srcoffset + sum(c .* srcstrides)` / `dstoffset + sum(c .* dststrides)`, `0 <= c[g] < dims[g]`.
# Strides are the tensor's own real strides (never derived from extents); `tilecounts[g] =
# cld(dims[g], tile[g])`. Plain integers only, so this is `isbits` and uploads to a GPU as-is.
struct BatchedPermuteDesc{N}
    dims::NTuple{N, Int}
    dststrides::NTuple{N, Int}
    srcstrides::NTuple{N, Int}
    tilecounts::NTuple{N, Int}
    dstoffset::Int
    srcoffset::Int
end

# N0 = batch rank, N = reduced rank (>= 1); no tensor shape or batch length enters the type.
# `descs`/`cpublocks` are never mutated in place once stored (they are only filled while local
# to the function building them). `_rebuild_descriptors` and the GPU binding cache both rely
# on this to hand back the same vectors, or the same plan object, when nothing changed.
struct BatchedPermutePlan{N0, N, T, I}
    perm::NTuple{N0, Int}
    srcorder::NTuple{N0, Int}         # heuristic source-axis order (`_infer_order`)
    groups::NTuple{N0, Int}           # reduced axis (1:N) of each srcorder position; 0 = dropped
    dstseq::NTuple{N, Int}            # heuristic destination order, in reduced labels
    family::BatchedPermuteFamily
    strategy::BatchedPermuteStrategy  # resolved once, at plan time
    tile::NTuple{N, Int}              # batch-wide tile shape H
    srclabels::NTuple{N, Int}         # original source axis of each reduced axis (0 = dummy)
    dstlabels::NTuple{N, Int}         # original destination axis of each reduced axis (0 = dummy)
    descs::Vector{BatchedPermuteDesc{N}}
    cpublocks::Vector{NTuple{N, Int}} # CPU-only sub-tile blocking per tensor (empty on GPU)
    prefix::Vector{Int}               # exclusive prefix sum of per-tensor tile counts, length B+1
    totaltiles::Int
    totalelements::Int
    deviceid::I
    devcache::Base.RefValue{Any}      # extension-owned slot for cached device-side state
end

# Hooks a GPU extension overrides for its own device-identity type; CPU is `deviceid === nothing`.
_batched_deviceid(::StridedView) = nothing

# CPU has a single execution strategy, so only `BP_AUTO` is legal; anything else asks for a GPU
# strategy on non-GPU arrays and is rejected rather than ignored.
_batched_resolve_strategy(::Nothing, requested::BatchedPermuteStrategy, ::BatchedPermuteFamily, ::Type) =
    requested === BP_AUTO ? BP_AUTO :
    throw(ArgumentError("batched_permutedims!: strategy=$requested applies only to GPU-backed batches"))

# CPU tile shape: about `totalelements / (nthreads * 8)` elements per tile (clamped), filled
# greedily from the axis fastest on both sides (else the source-fastest axis), then in axis
# order. `strategy`, `dims` and `T` are unused here; they exist for the GPU override.
function _fillorder(qt::NTuple{N, Int}) where {N}
    order = Int[1]
    qt[1] != 1 && push!(order, qt[1])
    for g in 1:N
        g in order || push!(order, g)
    end
    return order
end

function _batched_tileshape(
        ::Nothing, ::BatchedPermuteStrategy, ::BatchedPermuteFamily,
        qt::NTuple{N, Int}, maxdims::NTuple{N, Int}, ::Vector{NTuple{N, Int}}, ::Type,
        totalelements::Int, nthreads::Int
    ) where {N}
    cap = clamp(cld(totalelements, nthreads * 8), 1 << 10, 1 << 16)
    H = fill(1, N)
    p = 1
    for g in _fillorder(qt)
        H[g] = clamp(cap ÷ p, 1, max(maxdims[g], 1))
        p *= H[g]
    end
    return ntuple(i -> H[i], N)
end

_svtype(::Type{<:StridedView{T}}) where {T} = T
_cprod(t::Tuple) = foldl(Base.checked_mul, t; init = 1)

# Common storage order per side: a scheduling heuristic only, never used for addressing (that
# always uses each tensor's real strides). Takes the stride order of the first tensor with no
# singleton axis (unambiguous); otherwise `1:N0`, which is always safe, just unoptimized.
function _infer_order(views::Vector{<:StridedView}, N0::Int)
    for v in views
        sz = size(v)
        (prod(sz) == 0 || any(==(1), sz)) && continue
        st = strides(v)
        return Tuple(sort!(collect(1:N0); by = k -> st[k]))
    end
    return ntuple(identity, N0)
end

# Axis reduction: an axis with extent 1 in every tensor is dropped. Axes are never merged --
# fusing adjacent axes needs `stride[i] == stride[i-1] * extent[i-1]`, which is neither assumed
# nor checked. `q[j]` = heuristic-order source position filling destination position j.
# Returns (groups, qt, N, keep): `groups[i]` = reduced axis of heuristic position i (0 =
# dropped), `qt` = `q` relabeled to reduced axes, `keep[g]` = position that became reduced axis g.
function _collapse(nn::Vector{NTuple{N0, Int}}, q::NTuple{N0, Int}) where {N0}
    B = length(nn)
    dropped = ntuple(i -> all(b -> nn[b][i] == 1, 1:B), N0)
    keep = Int[i for i in 1:N0 if !dropped[i]]
    Nsurv = length(keep)
    relabel = Dict(i => k for (k, i) in enumerate(keep))
    N = max(Nsurv, 1)
    groups = ntuple(i -> dropped[i] ? 0 : relabel[i], N0)
    qt = Nsurv == 0 ? (1,) : Tuple(relabel[v] for v in q if !dropped[v])
    return groups, qt, N, keep
end

# Reduced extents of one tensor under a fixed `groups` (overflow-checked products; 1 for the
# dummy axis). `N` is a runtime `Int` and the `@inline` is load-bearing: inlined where `N` is a
# static plan parameter the `ntuple` folds to a fixed tuple with no allocation, whereas out of
# line it would return an abstract tuple and box it on every call.
@inline function _groupdims(nn_b::NTuple{N0, Int}, groups::NTuple{N0, Int}, N::Int) where {N0}
    return ntuple(N) do g
        p = 1
        for i in 1:N0
            groups[i] == g && (p = Base.checked_mul(p, nn_b[i]))
        end
        p
    end
end

# Original source/destination axis feeding each reduced axis; 0 marks the dummy axis of an
# all-dropped batch. `y[d] = x[perm[d]]`, so source label `s` lands on destination axis `invp[s]`.
function _axislabels(sigma::NTuple{N0, Int}, invp::NTuple{N0, Int}, keep::Vector{Int}, N::Int) where {N0}
    srclabels = ntuple(g -> g <= length(keep) ? sigma[keep[g]] : 0, N)
    dstlabels = ntuple(g -> srclabels[g] == 0 ? 0 : invp[srclabels[g]], N)
    return srclabels, dstlabels
end

# Each tensor's own strides at the labeled axes, nothing computed from extents; a dummy axis
# (label 0) gets stride 0 since its only coordinate is 0.
function _realstrides(
        sview::StridedView, dview::StridedView,
        srclabels::NTuple{N, Int}, dstlabels::NTuple{N, Int}
    ) where {N}
    sst = strides(sview)
    dst = strides(dview)
    srcstrides = ntuple(g -> srclabels[g] == 0 ? 0 : sst[srclabels[g]], N)
    dststrides = ntuple(g -> dstlabels[g] == 0 ? 0 : dst[dstlabels[g]], N)
    return srcstrides, dststrides
end

@inline function _make_desc(
        dview::StridedView, sview::StridedView, dims_b::NTuple{N, Int},
        srclabels::NTuple{N, Int}, dstlabels::NTuple{N, Int}, H::NTuple{N, Int}
    ) where {N}
    srcstrides_b, dststrides_b = _realstrides(sview, dview, srclabels, dstlabels)
    return BatchedPermuteDesc{N}(
        dims_b, dststrides_b, srcstrides_b, map(cld, dims_b, H), offset(dview), offset(sview)
    )
end

# CPU-only cache blocking of one tile, via Strided's existing `_computeblocks` heuristic; a pure
# function of the descriptor, `H` and `sizeof(T)`, so an unchanged descriptor implies an
# unchanged blocking.
function _cpuoblock(desc::BatchedPermuteDesc{N}, H::NTuple{N, Int}, sizeofT::Int) where {N}
    bytestrides = (sizeofT .* desc.dststrides, sizeofT .* desc.srcstrides)
    costs = _computecosts((desc.dststrides, desc.srcstrides))
    strideorders = (indexorder(desc.dststrides), indexorder(desc.srcstrides))
    return _computeblocks(min.(H, desc.dims), costs, bytestrides, strideorders)
end

# --- validation shared by fresh planning and plan reuse ---

function _check_deviceid(views, devid)
    for v in views
        isequal(_batched_deviceid(v), devid) ||
            throw(ArgumentError("batched_permutedims!: mixed backends/devices"))
    end
    return nothing
end

function _validate_deviceid(dviews, sviews)
    devid = !isempty(dviews) ? _batched_deviceid(dviews[1]) :
        !isempty(sviews) ? _batched_deviceid(sviews[1]) : nothing
    _check_deviceid(dviews, devid)
    _check_deviceid(sviews, devid)
    return devid
end

function _validate_sizes(dviews, sviews, perm::NTuple{N0, Int}) where {N0}
    for b in 1:length(dviews)
        szd = size(dviews[b])
        szs = size(sviews[b])
        for j in 1:N0
            szd[j] == szs[perm[j]] ||
                throw(DimensionMismatch("batched_permutedims!: size mismatch between dsts[$b] and srcs[$b]"))
        end
    end
    return nothing
end

# Not `map(StridedView, xs)`: that infers as `Union{Vector{Any}, Vector{StridedView{...}}}`
# (collect's type widening) even for a concrete `eltype(xs)`, turning every downstream loop
# into dynamic dispatch. Fixing `V` up front via `promote_op` behind a function barrier gives a
# concrete `Vector{V}`; a non-concrete `eltype(xs)` keeps the `map` path (run-time narrowing).
_normview_typed(::Type{V}, xs) where {V} = V[StridedView(x) for x in xs]
function _normview(xs::AbstractVector)
    V = Base.promote_op(StridedView, eltype(xs))
    isconcretetype(V) && return _normview_typed(V, xs)
    isempty(xs) && return Vector{V}()
    return map(StridedView, xs)
end

function _check_rank(views, N0::Int)
    for v in views
        ndims(v) == N0 || throw(DimensionMismatch("batched_permutedims!: rank mismatch"))
    end
    return nothing
end

# This API only moves raw bits, so a view carrying `conj`/`adjoint`/... must be rejected.
function _check_op(views)
    for v in views
        v.op === identity || throw(ArgumentError("batched_permutedims!: only identity ops are supported"))
    end
    return nothing
end

function _normalize_and_check(dsts::AbstractVector, srcs::AbstractVector, N0::Int)
    dviews = _normview(dsts)
    sviews = _normview(srcs)
    isconcretetype(eltype(dviews)) &&
        isconcretetype(eltype(sviews)) ||
        throw(ArgumentError("batched_permutedims!: mixed parent array types are not supported"))
    _check_rank(dviews, N0)
    _check_rank(sviews, N0)
    T = _svtype(eltype(dviews))
    T === _svtype(eltype(sviews)) ||
        throw(ArgumentError("batched_permutedims!: dsts and srcs must share one element type"))
    (isbitstype(T) && sizeof(T) > 0) ||
        throw(ArgumentError("batched_permutedims!: element type must be a nonzero-size isbits type"))
    _check_op(dviews)
    _check_op(sviews)
    return dviews, sviews, T
end

function _prepare(dsts::AbstractVector, srcs::AbstractVector, N0::Int)
    Base.require_one_based_indexing(dsts, srcs)
    length(dsts) == length(srcs) ||
        throw(DimensionMismatch("batched_permutedims!: dsts and srcs must have equal length"))
    return _normalize_and_check(dsts, srcs, N0)
end

_incompatible() = ArgumentError("batched_permutedims!: arrays are incompatible with this plan")

# --- per-call coefficients ---

# The one formula every executor uses for `beta != 0`; `beta == 0` never reads `y` at all (so a
# NaN-poisoned or uninitialized destination stays clean) and computes plain `a * x`.
@inline _batched_axpby(a, x, b, y) = a * x + b * y

# Both omitted -> `nothing` (the unscaled path, unchanged); otherwise a `(Vector{T}, Vector{T})`
# pair. That is a 2-valued compile-time distinction on `Nothing` (as `_mapreduce_kernel!` does
# for `op`), deliberately not a `Val`. Coefficients are converted to `T` once here, so a complex
# coefficient with nonzero imaginary part on a real batch throws `InexactError` from `convert`.
_batched_coeffs(::Nothing, ::Nothing, ::Int, ::Type) = nothing
function _batched_coeffs(alpha, beta, B::Int, ::Type{T}) where {T}
    T <: Number || throw(ArgumentError("batched_permutedims!: alpha/beta require a Number element type, got $T"))
    a = alpha === nothing ? ones(T, B) : _batched_coeffvec(alpha, B, T)
    b = beta === nothing ? zeros(T, B) : _batched_coeffvec(beta, B, T)
    return (a, b)
end
function _batched_coeffvec(x::AbstractVector, B::Int, ::Type{T}) where {T}
    length(x) == B || throw(DimensionMismatch("batched_permutedims!: alpha/beta must have length(dsts) entries"))
    return convert(Vector{T}, x)
end

# --- fresh plan ---

function _plan_impl(
        dsts::AbstractVector, srcs::AbstractVector, perm::NTuple{N0, Int};
        strategy::BatchedPermuteStrategy = BP_AUTO
    ) where {N0}
    isperm(perm) || throw(ArgumentError("batched_permutedims!: perm is not a valid permutation"))
    dviews, sviews, T = _prepare(dsts, srcs, N0)
    deviceid = _validate_deviceid(dviews, sviews)
    _validate_sizes(dviews, sviews, perm)
    sigma = _infer_order(sviews, N0)
    tau = _infer_order(dviews, N0)
    return _build_descriptors(dviews, sviews, perm, sigma, tau, deviceid, T, strategy)
end

function _build_descriptors(
        dviews::Vector{<:StridedView}, sviews::Vector{<:StridedView},
        perm::NTuple{N0, Int}, sigma::NTuple{N0, Int}, tau::NTuple{N0, Int},
        deviceid, ::Type{T}, strategy::BatchedPermuteStrategy
    ) where {N0, T}
    B = length(dviews)
    invsigma = invperm(sigma)
    invp = invperm(perm)
    q = ntuple(j -> invsigma[perm[tau[j]]], N0)
    nn = [ntuple(i -> size(sviews[b])[sigma[i]], N0) for b in 1:B]
    groups, qt, N, keep = _collapse(nn, q)
    srclabels, dstlabels = _axislabels(sigma, invp, keep, N)
    dimsvec = Vector{NTuple{N, Int}}(undef, B)   # typed explicitly: `N` is a runtime value here
    for b in 1:B
        dimsvec[b] = _groupdims(nn[b], groups, N)
    end
    maxdims = isempty(dimsvec) ? ntuple(_ -> 0, N) : reduce((a, c) -> map(max, a, c), dimsvec)
    totalelements = foldl((s, d) -> Base.checked_add(s, _cprod(d)), dimsvec; init = 0)
    family = N == 1 ? BP_COPY : (qt[1] == 1 ? BP_PAYLOAD : BP_TRANSPOSE)
    resolved = _batched_resolve_strategy(deviceid, strategy, family, T)
    H = _batched_tileshape(
        deviceid, resolved, family, qt, maxdims, dimsvec, T, totalelements, get_num_threads()
    )
    iscpu = deviceid === nothing
    descs = Vector{BatchedPermuteDesc{N}}(undef, B)
    cpublocks = iscpu ? Vector{NTuple{N, Int}}(undef, B) : NTuple{N, Int}[]
    prefix = Vector{Int}(undef, B + 1)
    prefix[1] = 0
    for b in 1:B
        desc = descs[b] = _make_desc(dviews[b], sviews[b], dimsvec[b], srclabels, dstlabels, H)
        prefix[b + 1] = Base.checked_add(prefix[b], _cprod(desc.tilecounts))
        iscpu && (cpublocks[b] = _cpuoblock(desc, H, sizeof(T)))
    end
    return BatchedPermutePlan{N0, N, T, typeof(deviceid)}(
        perm, sigma, groups, qt, family, resolved, H, srclabels, dstlabels,
        descs, cpublocks, prefix, prefix[B + 1], totalelements,
        deviceid, Base.RefValue{Any}(nothing)
    )
end

# --- plan reuse ---

# Reuse requires matching shapes only (tile counts/prefix sums depend on them); strides and
# offsets are re-read from the arrays passed in. Order inference, collapsing, tile shape and
# strategy are trusted from the plan. Fast path: every descriptor is recomputed and compared
# bitwise (`BatchedPermuteDesc` is `isbits`); while all match, the cached vectors are reused and
# the same plan object is returned with no allocation. From the first differing tensor on, fresh
# `descs` (and, on CPU, `cpublocks`) vectors are built, copying the unchanged prefix. Sound only
# because `plan.descs`/`plan.cpublocks` are never mutated in place anywhere.
function _rebuild_descriptors(
        dviews::Vector{<:StridedView}, sviews::Vector{<:StridedView},
        plan::BatchedPermutePlan{N0, N, T, I}
    ) where {N0, N, T, I}
    B = length(dviews)
    olddescs = plan.descs
    oldblocks = plan.cpublocks
    B == length(olddescs) || throw(_incompatible())
    descs = olddescs          # replaced by a fresh vector at the first differing tensor
    cpublocks = oldblocks     # likewise (CPU plans only)
    iscpu = plan.deviceid === nothing
    for b in 1:B
        nn_b = ntuple(i -> size(sviews[b])[plan.srcorder[i]], N0)
        dims_b = _groupdims(nn_b, plan.groups, N)
        dims_b == olddescs[b].dims || throw(_incompatible())
        desc_b = _make_desc(dviews[b], sviews[b], dims_b, plan.srclabels, plan.dstlabels, plan.tile)
        if descs === olddescs
            desc_b == olddescs[b] && continue
            descs = copyto!(Vector{BatchedPermuteDesc{N}}(undef, B), 1, olddescs, 1, b - 1)
            iscpu && (cpublocks = copyto!(Vector{NTuple{N, Int}}(undef, B), 1, oldblocks, 1, b - 1))
        end
        descs[b] = desc_b
        iscpu && (cpublocks[b] = _cpuoblock(desc_b, plan.tile, sizeof(T)))
    end
    descs === olddescs && return plan
    return BatchedPermutePlan{N0, N, T, I}(
        plan.perm, plan.srcorder, plan.groups, plan.dstseq, plan.family, plan.strategy,
        plan.tile, plan.srclabels, plan.dstlabels, descs, cpublocks, plan.prefix,
        plan.totaltiles, plan.totalelements, plan.deviceid, plan.devcache
    )
end

# --- public API ---

"""
    Strided.plan_batched_permutedims(dsts::AbstractVector, srcs::AbstractVector, perm;
                                      strategy::BatchedPermuteStrategy = BP_AUTO) -> BatchedPermutePlan

Plan a batched out-of-place permutation `dsts[b] = permutedims(srcs[b], perm)` for every `b`.
The plan can be passed to `batched_permutedims!` any number of times, including with different
arrays of the same shapes.

`strategy` is resolved once here; the plan's `.strategy` field reports the resolved value (never
`BP_AUTO`, except on CPU-backed batches, which have a single strategy and only accept `BP_AUTO`;
anything else is an error there). See `BatchedPermuteStrategy`.

Addressing uses each array's own strides, so non-dense and negative-stride views are supported.
There is no aliasing check: overlapping destinations, or a destination overlapping a source, is
undefined behavior. Sources may alias each other.
"""
function plan_batched_permutedims(
        dsts::AbstractVector, srcs::AbstractVector, perm::NTuple{N0, Int};
        strategy::BatchedPermuteStrategy = BP_AUTO
    ) where {N0}
    return _plan_impl(dsts, srcs, perm; strategy)
end
function plan_batched_permutedims(
        dsts::AbstractVector, srcs::AbstractVector, perm::AbstractVector{<:Integer};
        strategy::BatchedPermuteStrategy = BP_AUTO
    )
    return _plan_impl(dsts, srcs, ntuple(i -> Int(perm[i]), length(perm)); strategy)
end

"""
    Strided.batched_permutedims!(dsts::AbstractVector, srcs::AbstractVector, perm;
                                  strategy::BatchedPermuteStrategy = BP_AUTO,
                                  alpha = nothing, beta = nothing) -> dsts
    Strided.batched_permutedims!(dsts::AbstractVector, srcs::AbstractVector, plan::BatchedPermutePlan;
                                  alpha = nothing, beta = nothing) -> dsts

Batched out-of-place permutation `dsts[b] .= permutedims(srcs[b], perm)` for every `b`, or, with
coefficients, `dsts[b] .= alpha[b] .* permutedims(srcs[b], perm) .+ beta[b] .* dsts[b]`. The
`perm` form is `plan_batched_permutedims` (forwarding `strategy`) followed by the `plan` form,
which takes no `strategy` keyword since the plan carries its resolved one.

`alpha` and `beta` are optional per-tensor coefficient vectors of length `length(dsts)`. An
omitted `alpha` means all ones, an omitted `beta` all zeros, and omitting both is the plain copy
with no arithmetic at all. Coefficients are converted to the batch's element type `T` (which
must then be a `Number`): a real coefficient on a complex batch becomes `a + 0im`, and a complex
coefficient with a nonzero imaginary part on a real batch throws an `InexactError`. Wherever
`beta[b] == 0` the previous contents of `dsts[b]` are never read, so an uninitialized destination
is safe there; `alpha[b] == 1` gets no special treatment. The scaled result is ordinary
floating-point arithmetic in `T` and is not guaranteed to agree bitwise between the CPU and a
GPU backend (a GPU may fuse the multiply-add), unlike the coefficient-free copy, which moves bits.

On a GPU with `beta[b] != 0` for a `BP_TRANSPOSE` batch, `strategy = BP_GROUPTILE` reads the old
destination through shared memory and is substantially faster than the `BP_AUTO`/`BP_ELEMENTWISE`
default, which reads it through the same strided access pattern as the (already slower) transpose
store.

Not safe for aliased inputs: destinations must be pairwise disjoint and disjoint from every
source, or the result is unspecified; this is never checked. Sharing one plan across concurrent
tasks with *different* arrays on a GPU backend is not thread-safe (the plan's device cache is
mutated in place on reuse); concurrent reuse with the same arrays is safe.
"""
function batched_permutedims!(
        dsts::AbstractVector, srcs::AbstractVector, perm; strategy::BatchedPermuteStrategy = BP_AUTO,
        alpha::Union{Nothing, AbstractVector} = nothing, beta::Union{Nothing, AbstractVector} = nothing
    )
    plan = plan_batched_permutedims(dsts, srcs, perm; strategy)
    return batched_permutedims!(dsts, srcs, plan; alpha, beta)
end

function batched_permutedims!(
        dsts::AbstractVector, srcs::AbstractVector, plan::BatchedPermutePlan{N0, N, T, I};
        alpha::Union{Nothing, AbstractVector} = nothing, beta::Union{Nothing, AbstractVector} = nothing
    ) where {N0, N, T, I}
    dviews, sviews, Tact = _prepare(dsts, srcs, N0)
    Tact === T || throw(_incompatible())
    coeffs = _batched_coeffs(alpha, beta, length(dviews), T)
    deviceid = _validate_deviceid(dviews, sviews)
    isequal(deviceid, plan.deviceid) || throw(_incompatible())
    _validate_sizes(dviews, sviews, plan.perm)
    newplan = _rebuild_descriptors(dviews, sviews, plan)
    newplan.totaltiles == 0 || _batched_permute!(newplan, dviews, sviews, coeffs)
    return dsts
end

# `_batched_permute!(plan, dst::Vector{<:StridedView}, src::Vector{<:StridedView}, coeffs)` is
# defined in `batched_permutedims_cpu.jl`; the GPU extension adds a method for its own view
# subtype. `coeffs` is `nothing` or the `(alpha, beta)` pair from `_batched_coeffs`.
