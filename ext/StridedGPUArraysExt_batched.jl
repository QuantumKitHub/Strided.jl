# GPU executor: `_batched_permute!` for `GPUStridedView`s, plus the planner hooks. ONE kernel
# launch per batch: descriptors, tile-id prefix sums and base addresses are uploaded as device
# arrays (cached per plan), and every work item locates its own tensor and tile on the device.

import Strided: BatchedPermuteDesc, BatchedPermuteFamily, BatchedPermutePlan, BatchedPermuteStrategy
import Strided: BP_COPY, BP_PAYLOAD, BP_TRANSPOSE, BP_AUTO, BP_ELEMENTWISE, BP_THREADTILE, BP_GROUPTILE
import Strided: _batched_deviceid, _batched_tileshape, _batched_resolve_strategy, _tile_offsets, _batched_axpby
using GPUArrays.KernelAbstractions: @localmem, @synchronize

# `BP_THREADTILE`: elements per thread and along which axis (`:dstfast`/`:srcfast`); `Ref`s so a
# benchmark can flip them at runtime without recompiling anything.
const _BP_THREADTILE_K = Ref(8)
const _BP_THREADTILE_ALONG = Ref(:dstfast)

# `BP_GROUPTILE`: an EDGE x EDGE tile transposed through local memory by EDGE*ROWS threads
# (EDGE÷ROWS rows each). Each entry is compiled separately via `Val(edge)`/`Val(rows)` -- the ONE
# sanctioned exception to "no shape-like quantity enters a type domain", bounded by the menu size
# (<= 3, asserted). Nothing derived from a tensor's shape, stride, offset, batch length or tile
# count may join it or become a `Val` elsewhere. Ordered by decreasing edge; entry 1 is the
# default and wins ties. ROWS must divide EDGE, or the load/store loops silently skip tile rows.
const _BP_GROUPTILE_GEOMETRIES = ((edge = 32, rows = 4), (edge = 16, rows = 4))
const _BP_GROUPTILE_PAD = 1               # +1 column against shared-memory bank conflicts
const _BP_GROUPTILE_META_LEN = 6
@assert _BP_GROUPTILE_META_LEN == 6  # kernel uses literal meta[1..6]; lowering this silently
# leaves those indices out of bounds (a wrong result, not an error)
const _BP_GROUPTILE_SHMEM_BUDGET = 32768  # tile bytes only; well under CUDA's 48 KiB for Metal/ROCm
@assert _BP_GROUPTILE_SHMEM_BUDGET + _BP_GROUPTILE_META_LEN * sizeof(Int) <= 49152
@assert 1 <= length(_BP_GROUPTILE_GEOMETRIES) <= 3
@assert all(g -> g.edge >= 1 && g.rows >= 1 && g.edge % g.rows == 0, _BP_GROUPTILE_GEOMETRIES)
@assert all(
    i -> _BP_GROUPTILE_GEOMETRIES[i].edge > _BP_GROUPTILE_GEOMETRIES[i + 1].edge,
    1:(length(_BP_GROUPTILE_GEOMETRIES) - 1)
)

_batched_grouptile_tilebytes(::Type{T}, edge::Int) where {T} =
    sizeof(T) * (edge + _BP_GROUPTILE_PAD) * edge
_batched_grouptile_fits(::Type{T}, edge::Int) where {T} =
    _batched_grouptile_tilebytes(T, edge) <= _BP_GROUPTILE_SHMEM_BUDGET

# Tile-slot utilization of `edge`: real elements / scheduled edge^2 thread slots over the batch.
# Active axes are 1 and `c`; the other extents multiply the tile count.
function _batched_grouptile_utilization(dims, c::Int, edge::Int)
    elements = 0
    slots = 0
    for d in dims
        ntiles = 1
        for g in eachindex(d)
            ntiles *= (g == 1 || g == c) ? cld(d[g], edge) : d[g]
        end
        slots += ntiles * edge * edge
        elements += prod(d)
    end
    return slots == 0 ? 1.0 : elements / slots
end

# Menu index with the highest utilization among entries whose tile fits `T`; a tie keeps the
# earlier (larger) entry. Some entry always fits when this runs (the resolver admits
# `BP_GROUPTILE` only then), so the throw marks an invariant rather than a caught path.
function _batched_grouptile_geometry(dims, c::Int, ::Type{T}) where {T}
    best = 0
    bestu = -1.0
    for (i, geom) in enumerate(_BP_GROUPTILE_GEOMETRIES)
        _batched_grouptile_fits(T, geom.edge) || continue
        u = _batched_grouptile_utilization(dims, c, geom.edge)
        if u > bestu
            best, bestu = i, u
        end
    end
    best == 0 && throw(ArgumentError(
        "batched_permutedims!: internal invariant violated: no BP_GROUPTILE geometry fits element type $T"
    ))
    return best
end

# `BP_AUTO` -> the elementwise baseline (unchanged default). `BP_GROUPTILE` needs the transpose
# family (N >= 2, `qt[1] != 1`) and an element type some menu tile fits; otherwise it falls back
# to `BP_ELEMENTWISE` deliberately -- not an error, inspectable via `plan.strategy`. `any` fits
# <=> the smallest edge fits, and the tile-shape hook selects only among fitting entries.
function _batched_resolve_strategy(
        ::KernelAbstractions.Backend, requested::BatchedPermuteStrategy,
        family::BatchedPermuteFamily, ::Type{T}
    ) where {T}
    requested === BP_AUTO && return BP_ELEMENTWISE
    requested === BP_GROUPTILE || return requested
    fits = any(geom -> _batched_grouptile_fits(T, geom.edge), _BP_GROUPTILE_GEOMETRIES)
    return (family === BP_TRANSPOSE && fits) ? BP_GROUPTILE : BP_ELEMENTWISE
end

# Distinguishes backends only, not devices within one backend (KA backends carry no device index).
_batched_deviceid(a::GPUStridedView) = KernelAbstractions.get_backend(parent(a))

# ELEMENTWISE: tile 1 everywhere (one element per thread). THREADTILE: `k` elements along one
# axis -- `:srcfast` or BP_COPY: axis 1; `:dstfast`: `qt[1]` for transposes, axis 2 for payload
# (axis 1 is already the shared contiguous run) -- clamped to that axis's largest extent.
# GROUPTILE: a menu edge on axes 1 and `c = qt[1]` (distinct, since only the transpose family
# gets here), 1 elsewhere; the tile's edge is how the chosen geometry reaches the executor.
function _batched_tileshape(
        ::KernelAbstractions.Backend, strategy::BatchedPermuteStrategy, family::BatchedPermuteFamily,
        qt::NTuple{N, Int}, maxdims::NTuple{N, Int}, dims::Vector{NTuple{N, Int}}, ::Type{T},
        ::Int, ::Int
    ) where {N, T}
    if strategy === BP_THREADTILE
        k = _BP_THREADTILE_K[]
        along = _BP_THREADTILE_ALONG[]
        along in (:dstfast, :srcfast) ||
            throw(ArgumentError("_BP_THREADTILE_ALONG[] must be :dstfast or :srcfast, got $along"))
        axis = (along === :srcfast || family === BP_COPY) ? 1 : (family === BP_TRANSPOSE ? qt[1] : 2)
        kk = clamp(k, 1, max(maxdims[axis], 1))
        return ntuple(g -> g == axis ? kk : 1, N)
    elseif strategy === BP_GROUPTILE
        c = qt[1]
        edge = _BP_GROUPTILE_GEOMETRIES[_batched_grouptile_geometry(dims, c, T)].edge
        return ntuple(g -> (g == 1 || g == c) ? edge : 1, N)
    else # BP_ELEMENTWISE
        return ntuple(Returns(1), N)
    end
end

# --- device upload ---

# Descriptors hold plain integers (no pointers), so they upload like any array of numbers.
function _batched_upload(proto, x::Vector)
    d = similar(proto, eltype(x), (length(x),))
    copyto!(d, x)
    return d
end

# Per-tensor base addresses: `B` is a runtime value, so the parents go into one uploaded array.
# Preferred element: `KernelAbstractions.argconvert(kernel!, parent)`, the backend's own
# indexable device-array struct. Where that is not `isbits` (the JLArrays backend) fall back to
# raw `UInt64` addresses, reinterpreted as `Ptr{T}` in the kernel. Detected by trying, not by name.
@inline function _batched_try_argconvert(kernel!, p)
    local adapted
    try
        adapted = KernelAbstractions.argconvert(kernel!, p)
    catch
        return nothing
    end
    return isbitstype(typeof(adapted)) ? adapted : nothing
end

function _batched_convert_bases(kernel!, parents::Vector)
    first_base = _batched_try_argconvert(kernel!, parents[1])
    if first_base !== nothing
        DT = typeof(first_base)
        return DT[something(_batched_try_argconvert(kernel!, p))::DT for p in parents]
    else
        return UInt64[UInt64(UInt(pointer(p))) for p in parents]
    end
end

@inline _gpu_getbase(base::UInt64, ::Val{T}, i::Int) where {T} =
    unsafe_load(reinterpret(Ptr{T}, base), i)
@inline _gpu_setbase!(base::UInt64, ::Val{T}, i::Int, v) where {T} =
    unsafe_store!(reinterpret(Ptr{T}, base), v, i)
@inline _gpu_getbase(base, ::Val, i::Int) = (@inbounds base[i])
@inline _gpu_setbase!(base, ::Val, i::Int, v) = (@inbounds base[i] = v; nothing)

# Per-tensor coefficients inside a kernel: `nothing` when the call has none (the store below is
# then exactly the plain store), else `(alpha[b], beta[b])`. `beta == 0` must not read `dst`.
@inline _gpu_coef(::Nothing, ::Nothing, ::Int) = nothing
@inline _gpu_coef(alpha, beta, b::Int) = (@inbounds (alpha[b], beta[b]))
@inline _gpu_put!(dbase, valT::Val, i::Int, v, ::Nothing) = _gpu_setbase!(dbase, valT, i, v)
@inline function _gpu_put!(dbase, valT::Val, i::Int, v, (a, b)::Tuple)
    w = iszero(b) ? a * v : _batched_axpby(a, v, b, _gpu_getbase(dbase, valT, i))
    return _gpu_setbase!(dbase, valT, i, w)
end

# --- device binding cache (`plan.devcache`) ---

# The upload is reused only if the device identity, EVERY parent's current `pointer(...)`, and
# the full host descriptors (dims, strides, offsets, tile counts -- plan reuse only requires
# matching shapes, so strides can differ at the same address) all still match. The per-call
# address read is mandatory, with no identity or length shortcut: `resize!` can move a buffer
# while keeping the same array object and length, and a stale binding would then read/write
# freed memory silently. Strong refs to the parents keep them alive while the binding is cached.
mutable struct _BatchedGPUBinding
    deviceid::Any
    srcaddrs::Vector{UInt}
    dstaddrs::Vector{UInt}
    hostdescs::Any            # `plan.descs` at upload time. On unchanged reuse this IS the live
    # vector, a valid snapshot only because plan descriptors are never mutated in place.
    descs::Any
    prefix::Any
    srcbases::Any
    dstbases::Any
    srcparents::Vector{Any}
    dstparents::Vector{Any}
end

function _batched_addrs_match(addrs::Vector{UInt}, parents::Vector)
    length(addrs) == length(parents) || return false
    for i in eachindex(parents)
        addrs[i] == UInt(pointer(parents[i])) || return false
    end
    return true
end

# Stops at the first mismatch but never skips a check class; allocates nothing on a match.
# `===` only shortcuts the `==` that would follow (reflexive for a vector of `isbits` descriptors).
function _batched_binding_matches(
        b::_BatchedGPUBinding, deviceid, srcparents::Vector, dstparents::Vector, hostdescs
    )
    isequal(b.deviceid, deviceid) || return false
    _batched_addrs_match(b.srcaddrs, srcparents) || return false
    _batched_addrs_match(b.dstaddrs, dstparents) || return false
    return b.hostdescs === hostdescs || b.hostdescs == hostdescs
end

_batched_addresses(parents::Vector) = UInt[UInt(pointer(p)) for p in parents]

# The binding is built for the kernel actually launched (`argconvert` takes the kernel); a plan's
# strategy is fixed, so a cached binding is only ever reused with the same kernel.
function _batched_get_binding!(
        plan::BatchedPermutePlan, kernel!, srcparents::Vector, dstparents::Vector
    )
    hostdescs = plan.descs
    cached = plan.devcache[]
    cached isa _BatchedGPUBinding &&
        _batched_binding_matches(cached, plan.deviceid, srcparents, dstparents, hostdescs) &&
        return cached
    proto = srcparents[1]
    binding = _BatchedGPUBinding(
        plan.deviceid, _batched_addresses(srcparents), _batched_addresses(dstparents), hostdescs,
        _batched_upload(proto, plan.descs), _batched_upload(proto, plan.prefix),
        _batched_upload(proto, _batched_convert_bases(kernel!, srcparents)),
        _batched_upload(proto, _batched_convert_bases(kernel!, dstparents)),
        Any[srcparents...], Any[dstparents...]
    )
    plan.devcache[] = binding
    return binding
end

# --- kernels ---

# `searchsortedlast` by hand: device code cannot use Base's generic implementation.
@inline function _batched_dev_searchsortedlast(prefix, u::Int)
    lo = 1
    hi = length(prefix)
    while lo < hi
        mid = (lo + hi + 1) >> 1
        @inbounds if prefix[mid] <= u
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

# Tensor index, descriptor and local tile id of global tile `u`.
@inline function _batched_dev_locate(descs, prefix, u::Int)
    b = _batched_dev_searchsortedlast(prefix, u)
    @inbounds return b, descs[b], u - prefix[b]
end

# One work item per global tile; shared by ELEMENTWISE and THREADTILE, which differ only in
# `tile`. `@index(Global, Cartesian)` with a 1-tuple ndrange: `Linear` fails on the JLArrays
# backend in this KA version (Strided's own GPU mapreduce kernel does the same). Work items past
# `ndrange` from workgroup padding are masked by KA's own `__validindex` guard.
@kernel function _batched_permute_gpu_kernel!(
        descs, prefix, dstbases, srcbases, tile::NTuple{N, Int}, valT::Val{T}, alpha, beta
    ) where {N, T}
    Idx = @index(Global, Cartesian)
    b, desc, l = _batched_dev_locate(descs, prefix, Idx[1] - 1)
    soff0, doff0, d = _tile_offsets(desc, tile, l)
    @inbounds sbase = srcbases[b]
    @inbounds dbase = dstbases[b]
    ab = _gpu_coef(alpha, beta, b)
    for I in CartesianIndices(map(Base.OneTo, d))
        so = soff0
        do_ = doff0
        for g in 1:N
            @inbounds so += (I[g] - 1) * desc.srcstrides[g]
            @inbounds do_ += (I[g] - 1) * desc.dststrides[g]
        end
        v = _gpu_getbase(sbase, valT, so + 1)
        _gpu_put!(dbase, valT, do_ + 1, v, ab)
    end
end

# `t[c]` for a runtime axis `c` WITHOUT indexing the tuple: a dynamic `getindex` in device code
# spills the whole tuple (and what it came from) to per-thread local memory -- measured as the
# single largest cost in the cooperative kernel. This unrolls to `N-1` branch-free selects.
# Defined for `1 <= c <= N`; out of range yields 0 (a wrong address, not an error), which is why
# the launch site checks the tile/`c` invariant before every launch.
@inline _batched_selaxis(t::Tuple, c::Int) = _batched_selaxis(t, c, 1)
@inline _batched_selaxis(t::Tuple, c::Int, g::Int) =
    ifelse(g == c, t[1], _batched_selaxis(Base.tail(t), c, g + 1))
@inline _batched_selaxis(::Tuple{}, ::Int, ::Int) = 0

# One workgroup of EDGE*ROWS threads per global tile, EDGE x EDGE on axes 1 (source-fastest)
# and `c` (destination-fastest). Reads coalesced along axis 1 into `lmem`, barrier, writes the
# tile transposed, coalesced along `c`. Partial tiles are guarded per load/store; the barrier
# never is. `EDGE`/`ROWS` come only from the menu entry matching `plan.tile` (launch site).
#
# Performance rules (both measured, not stylistic):
#   * never index a tuple with a runtime index in this body -- use `_batched_selaxis`;
#   * `g` is workgroup-uniform, so thread 1 publishes the six store-phase scalars into `meta`
#     before the barrier: one lookup + `_tile_offsets` per workgroup instead of two per thread.
# KernelAbstractions CPU-backend (JLArrays) rules, both load-bearing:
#   1. every `@localmem`/`@index` is a bare top-level `lhs = ...` statement, `@localmem` first;
#      nested in `if`/`let`/loops or larger expressions the CPU transform cannot re-splice it;
#   2. `@synchronize()` splits the body into separate work-item loops on the CPU, so no ordinary
#      local survives it -- only `@localmem` arrays and re-spliced `@index` values do (hence
#      `meta`, and the restated `l = @index(Local, Linear)` after the barrier).
@kernel function _batched_permute_gpu_grouptile_kernel!(
        descs, prefix, dstbases, srcbases, tile::NTuple{N, Int}, c::Int, valT::Val{T},
        ::Val{EDGE}, ::Val{ROWS}, alpha, beta
    ) where {N, T, EDGE, ROWS}
    lmem = @localmem T (EDGE + _BP_GROUPTILE_PAD, EDGE)
    meta = @localmem Int (_BP_GROUPTILE_META_LEN,)
    g = @index(Group, Linear)
    l = @index(Local, Linear)
    b, desc, lt = _batched_dev_locate(descs, prefix, g - 1)
    soff0, doff0, d = _tile_offsets(desc, tile, lt)
    d1 = d[1]
    dc = _batched_selaxis(d, c)
    s1 = desc.srcstrides[1]
    sc = _batched_selaxis(desc.srcstrides, c)
    @inbounds sbase = srcbases[b]
    if l == 1
        @inbounds meta[1] = b
        @inbounds meta[2] = d1
        @inbounds meta[3] = dc
        @inbounds meta[4] = desc.dststrides[1]
        @inbounds meta[5] = _batched_selaxis(desc.dststrides, c)
        @inbounds meta[6] = doff0
    end
    li = (l - 1) % EDGE + 1
    lj0 = (l - 1) ÷ EDGE + 1
    for k in 0:(EDGE ÷ ROWS - 1)
        lj = lj0 + k * ROWS
        if li <= d1 && lj <= dc
            @inbounds lmem[li, lj] = _gpu_getbase(sbase, valT, soff0 + (li - 1) * s1 + (lj - 1) * sc + 1)
        end
    end
    @synchronize()
    l = @index(Local, Linear)
    @inbounds b2 = meta[1]
    @inbounds d1b = meta[2]
    @inbounds dcb = meta[3]
    @inbounds t1 = meta[4]
    @inbounds tc = meta[5]
    @inbounds doff0 = meta[6]
    @inbounds dbase = dstbases[b2]
    ab = _gpu_coef(alpha, beta, b2)
    li = (l - 1) % EDGE + 1
    lj0 = (l - 1) ÷ EDGE + 1
    for k in 0:(EDGE ÷ ROWS - 1)
        lj = lj0 + k * ROWS
        if lj <= d1b && li <= dcb
            @inbounds v = lmem[lj, li]
            _gpu_put!(dbase, valT, doff0 + (lj - 1) * t1 + (li - 1) * tc + 1, v, ab)
        end
    end
end

# --- launch ---

# Every call must have finished moving data before returning (it owns the lifetime of the
# device metadata). The JLArrays backend has no `synchronize` method in this KA version but runs
# launches synchronously, so exactly that `MethodError` means "nothing to wait for"; anything
# else propagates.
function _batched_gpu_synchronize(backend)
    try
        KernelAbstractions.synchronize(backend)
    catch e
        e isa MethodError || rethrow()
    end
    return nothing
end

# `GC.@preserve` keeps the parents (and the per-call coefficient uploads in `args`) alive across
# launch + synchronize, on top of the binding's own strong references.
function _batched_launch!(plan::BatchedPermutePlan, kernel!, backend, srcparents, dstparents, args...; ndrange)
    binding = _batched_get_binding!(plan, kernel!, srcparents, dstparents)
    GC.@preserve srcparents dstparents args begin
        kernel!(binding.descs, binding.prefix, binding.dstbases, binding.srcbases, plan.tile, args...; ndrange)
        _batched_gpu_synchronize(backend)
    end
    return nothing
end

# Coefficients are per call, so they are uploaded per call and never enter the binding cache.
_batched_upload_coeffs(::Any, ::Nothing) = (nothing, nothing)
_batched_upload_coeffs(proto, (a, b)::Tuple) = (_batched_upload(proto, a), _batched_upload(proto, b))

function Strided._batched_permute!(
        plan::BatchedPermutePlan{N0, N, T}, dst::Vector{<:GPUStridedView}, src::Vector{<:GPUStridedView},
        coeffs
    ) where {N0, N, T}
    srcparents = parent.(src)
    dstparents = parent.(dst)
    backend = KernelAbstractions.get_backend(srcparents[1])
    alpha, beta = _batched_upload_coeffs(srcparents[1], coeffs)
    strategy = plan.strategy
    if strategy === BP_GROUPTILE
        # The tile must be exactly a menu edge on axes 1 and `c` (with `c != 1`), 1 elsewhere,
        # and that edge must fit `T`: `_batched_selaxis` would yield a wrong address, not an
        # error, and an unfitting edge would request local memory over budget. Checked, not assumed.
        c = plan.dstseq[1]
        gi = findfirst(geom -> geom.edge == plan.tile[1], _BP_GROUPTILE_GEOMETRIES)
        tileok = gi !== nothing && c != 1 && plan.tile[c] == plan.tile[1] &&
            all(g == 1 || g == c || plan.tile[g] == 1 for g in eachindex(plan.tile))
        tileok || throw(ArgumentError(
            "batched_permutedims!: internal invariant violated: BP_GROUPTILE plan has tile=$(plan.tile), dstseq=$(plan.dstseq)"
        ))
        geom = _BP_GROUPTILE_GEOMETRIES[gi]
        _batched_grouptile_fits(T, geom.edge) || throw(ArgumentError(
            "batched_permutedims!: internal invariant violated: BP_GROUPTILE tile edge $(geom.edge) does not fit element type $T"
        ))
        wgsize = geom.edge * geom.rows
        kernel! = _batched_permute_gpu_grouptile_kernel!(backend, (wgsize,))
        # `Val(geom.edge)`/`Val(geom.rows)`: `geom` is a menu entry, so at most one variant per entry.
        _batched_launch!(
            plan, kernel!, backend, srcparents, dstparents, c, Val(T), Val(geom.edge), Val(geom.rows),
            alpha, beta; ndrange = (plan.totaltiles * wgsize,)
        )
    elseif strategy === BP_ELEMENTWISE || strategy === BP_THREADTILE
        kernel! = _batched_permute_gpu_kernel!(backend)
        _batched_launch!(
            plan, kernel!, backend, srcparents, dstparents, Val(T), alpha, beta; ndrange = (plan.totaltiles,)
        )
    else
        throw(ArgumentError(
            "batched_permutedims!: internal invariant violated: unresolved GPU strategy $strategy at execution time"
        ))
    end
    return dst
end
