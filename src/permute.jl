"""
    permutedims!(C::StridedView, A::StridedView, p, α::Number, β::Number)

Compute `C = β * C + α * permutedims(A, p)` in a single pass, without reading `C` if `β`
is zero.
"""
function Base.permutedims!(
        C::StridedView{<:Any, N}, A::StridedView{<:Any, N}, p, α::Number, β::Number
    ) where {N}
    perm = _permtuple(p, Val(N))
    dims = size(C)
    TupleTools.getindices(size(A), perm) == dims ||
        throw(DimensionMismatch("destination size $dims does not match the permuted source"))
    any(iszero, dims) && return C
    sA = TupleTools.getindices(strides(A), perm)
    if !(_isdense_cpu(C) && _isdense_cpu(A)) ||
            prod(dims) * (sizeof(eltype(C)) + sizeof(eltype(A))) > BLOCKMEMORYSIZE
        arrays = (C, StridedView(parent(A), dims, sA, offset(A), A.op))
        if iszero(β)
            _mapreduce_order!(_Scale(α), nothing, nothing, dims, arrays)
        else
            _mapreduce_order!(_Scale(α), +, _Scale(β), dims, arrays)
        end
    elseif iszero(β)
        _permute_small!(_Scale(α), dims, (C, A), (strides(C), sA))
    else
        _permute_small!(_ScaleAdd(α, β), dims, (C, C, A), (strides(C), strides(C), sA))
    end
    return C
end

_permtuple(p::NTuple{N, Int}, ::Val{N}) where {N} =
    isperm(p) ? p : throw(ArgumentError("$p is not a valid permutation"))
_permtuple(p, ::Val{N}) where {N} = _permtuple(ntuple(i -> Int(p[i]), Val(N)), Val(N))

# the direct kernel call is only valid for arrays in CPU memory
@static if isdefined(Core, :Memory)
    _isdense_cpu(a::StridedView) = parent(a) isa Union{Array, Memory}
else
    _isdense_cpu(a::StridedView) = parent(a) isa Array
end

struct _Scale{T}
    α::T
end
(s::_Scale)(a) = s.α * a
struct _ScaleAdd{Tα, Tβ}
    α::Tα
    β::Tβ
end
(s::_ScaleAdd)(c, a) = s.β * c + s.α * a

# Arrays that fit in a block need no blocking or threading, and their loop order hardly
# matters: loop in the memory order of the destination and call the kernel directly.
function _permute_small!(f, dims::Dims{N}, arrays, allstrides) where {N}
    order = _sortperm_small(ntuple(i -> dims[i] == 1 ? typemax(Int) : abs(allstrides[1][i]), Val(N)))
    odims = TupleTools.getindices(dims, order)
    ostrides = map(s -> TupleTools.getindices(s, order), allstrides)
    kdims, kstrides = _fuse_compact(odims, ostrides)
    _mapreduce_kernel!(f, nothing, nothing, kdims, kdims, arrays, kstrides, map(offset, arrays))
    return nothing
end

# stable sort permutation for a short tuple; returns the identity without work if sorted
function _sortperm_small(keys::NTuple{N, Int}) where {N}
    issorted(keys) && return ntuple(identity, Val(N))
    return invperm(ntuple(i -> _stablerank(keys, i), Val(N)))
end
function _stablerank(keys::NTuple{N, Int}, i::Int) where {N}
    r = 1
    for j in 1:N
        r += (keys[j] < keys[i]) | ((keys[j] == keys[i]) & (j < i))
    end
    return r
end

# Fuse dimension `i` into `i - 1` when it is contiguous with it in all arrays, as
# `_fusedims`, but drop the fused dimensions instead of leaving singletons in their place.
# Whether a dimension fuses only depends on the original sizes and strides.
function _fuse_compact(dims::NTuple{N, Int}, allstrides) where {N}
    fuse = ntuple(i -> i > 1 && all(s -> s[i] == dims[i - 1] * s[i - 1], allstrides), Val(N))
    heads = ntuple(k -> _kthhead(fuse, k), Val(N))
    newdims = ntuple(k -> _headlength(dims, fuse, heads[k]), Val(N))
    newstrides = map(s -> ntuple(k -> heads[k] == 0 ? 0 : s[heads[k]], Val(N)), allstrides)
    return newdims, newstrides
end
function _kthhead(fuse::NTuple{N, Bool}, k::Int) where {N}
    for i in 1:N
        k -= !fuse[i]
        k == 0 && return i
    end
    return 0
end
function _headlength(dims::NTuple{N, Int}, fuse::NTuple{N, Bool}, h::Int) where {N}
    h == 0 && return 1
    d = dims[h]
    for j in (h + 1):N
        fuse[j] || break
        d *= dims[j]
    end
    return d
end
