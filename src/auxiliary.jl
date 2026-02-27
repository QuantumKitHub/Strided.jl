# Auxiliary routines
#---------------------
# Check whether p is a valid permutation of length N
_isperm(N::Integer, p::AbstractVector) = (length(p) == N && isperm(p))
_isperm(N::Integer, p::NTuple{M, Integer}) where {M} = (M == N && isperm(p))
_isperm(N::Integer, p) = false

# Compute the memory index given a list of cartesian indices and corresponding strides
@inline function _computeind(indices::NTuple{N, Int}, strides::NTuple{N, Int}) where {N}
    return (indices[1] - 1) * strides[1] + _computeind(tail(indices), tail(strides))
end
_computeind(indices::Tuple{}, strides::Tuple{}) = 1

# 'Simplify' the dimensions of a `StridedView` to represent it with the largest possible
# contiguous dimensions (thereby potentially fusing subsequent dimensions), without changing
# the order of the elements. For type stability, we do not remove dimensions but replace
# them with dimensions of size 1 and move all of those to the end.
_simplifydims(size::Tuple{}, strides::Tuple{}) = size, strides
_simplifydims(size::Dims{1}, strides::Dims{1}) = size, strides
function _simplifydims(size::Dims{N}, strides::Dims{N}) where {N}
    tailsize, tailstrides = _simplifydims(tail(size), tail(strides))
    if size[1] == 1
        return (tailsize..., 1), (tailstrides..., 1)
    elseif size[1] * strides[1] == tailstrides[1]
        return (size[1] * tailsize[1], tail(tailsize)..., 1),
            (strides[1], tail(tailstrides)..., 1)
    else
        return (size[1], tailsize...), (strides[1], tailstrides...)
    end
end

# 'Normalize' the strides of a `StridedView`, i.e. strides associated with dimensions of
# size 1 have no intrinsic meaning and can be changed arbitrarily. If one of the dimensions
# has size zero, then the whole array has length zero, and all strides are ambiguous. All
# ambiguous strides are set to 1.
function _normalizestrides(size::Dims{N}, strides::Dims{N}) where {N}
    for i in 1:N
        if size[i] == 1
            newstride = i == 1 ? 1 : strides[i - 1] * size[i - 1]
            strides = Base.setindex(strides, newstride, i)
        elseif size[i] == 0
            return (1, Base.front(cumprod(size))...)
        end
    end
    return strides
end

# 'Normalize' the layout of a DenseArray, in order to reduce the number of required
# specializations in functions.
@inline _normalizeparent(A) = A
@static if isdefined(Core, :Memory)
    @inline _normalizeparent(A::Array) = A.ref.mem
else
    @inline _normalizeparent(A::Array) = reshape(A, length(A))
end

# Auxiliary methods for `sview`
#------------------------------
# Compute the new dimensions of a strided view given the original size and the view slicing
# indices
@inline function _computeviewsize(oldsize::NTuple{N, Int}, I::NTuple{N, SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewsize(tail(oldsize), tail(I))
    elseif isa(I[1], Colon)
        return (oldsize[1], _computeviewsize(tail(oldsize), tail(I))...)
    else
        return (length(I[1]), _computeviewsize(tail(oldsize), tail(I))...)
    end
end
_computeviewsize(::Tuple{}, ::Tuple{}) = ()

# Compute the new strides of a (strided) view given the original strides and the view
# slicing indices
@inline function _computeviewstrides(
        oldstrides::NTuple{N, Int},
        I::NTuple{N, SliceIndex}
    ) where {N}
    if isa(I[1], Integer)
        return _computeviewstrides(tail(oldstrides), tail(I))
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computeviewstrides(tail(oldstrides), tail(I))...)
    else
        return (
            oldstrides[1] * step(I[1]),
            _computeviewstrides(tail(oldstrides), tail(I))...,
        )
    end
end
_computeviewstrides(::Tuple{}, ::Tuple{}) = ()

# Compute the additional offset of a (strided) view given the original strides and the view
# slicing indices
@inline function _computeviewoffset(
        strides::NTuple{N, Int},
        I::NTuple{N, SliceIndex}
    ) where {N}
    if isa(I[1], Colon)
        return _computeviewoffset(tail(strides), tail(I))
    else
        return (first(I[1]) - 1) * strides[1] + _computeviewoffset(tail(strides), tail(I))
    end
end
_computeviewoffset(::Tuple{}, ::Tuple{}) = 0

# Auxiliary methods for `sreshape`
#----------------------------------
# Compute the new strides of a (strided) reshape given the original strides and new and
# original sizes
_computereshapestrides(newsize::Tuple{}, oldsize::Tuple{}, strides::Tuple{}) = strides
function _computereshapestrides(
        newsize::Tuple{}, oldsize::Dims{N},
        strides::Dims{N}
    ) where {N}
    all(isequal(1), oldsize) || throw(DimensionMismatch())
    return ()
end
function _computereshapestrides(newsize::Dims, oldsize::Tuple{}, strides::Tuple{})
    all(isequal(1), newsize) || throw(DimensionMismatch())
    return newsize
end
function _computereshapestrides(newsize::Dims, oldsize::Dims{N}, strides::Dims{N}) where {N}
    d, r = divrem(oldsize[1], newsize[1])
    return if r == 0
        s1 = strides[1]
        if d == 1
            # not shrinking the following tuples helps type inference
            oldsize = (tail(oldsize)..., 1)
            strides = (tail(strides)..., 1)
            stail = _computereshapestrides(tail(newsize), oldsize, strides)
            return isnothing(stail) ? nothing : (s1, stail...)
        else
            oldsize = (d, tail(oldsize)...)
            strides = (newsize[1] * s1, tail(strides)...)
            stail = _computereshapestrides(tail(newsize), oldsize, strides)
            return isnothing(stail) ? nothing : (s1, stail...)
        end
    else
        if prod(newsize) != prod(oldsize)
            throw(DimensionMismatch())
        else
            return nothing
        end
    end
end
