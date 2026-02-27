################## StridedView.jl #
##################
# Defines the main type of this package and its functionality

# Preliminary:
#--------------
# Additional view flags and their transformation behaviour
const FN = typeof(identity)
const FC = typeof(conj)
const FA = typeof(adjoint)
const FT = typeof(transpose)
_conj(::FN) = conj
_conj(::FC) = identity
_conj(::FA) = transpose
_conj(::FT) = adjoint
_transpose(::FN) = transpose
_transpose(::FC) = adjoint
_transpose(::FA) = conj
_transpose(::FT) = identity
_adjoint(::FN) = adjoint
_adjoint(::FC) = transpose
_adjoint(::FA) = identity
_adjoint(::FT) = conj

# StridedView type definition
#-----------------------------
struct StridedView{T, N, A <: DenseArray, F <: Union{FN, FC, FA, FT}} <: AbstractArray{T, N}
    parent::A
    size::NTuple{N, Int}
    strides::NTuple{N, Int}
    offset::Int
    op::F
end

# Constructors
#--------------
function StridedView(
        parent::DenseArray,
        size::NTuple{N, Int} = size(parent),
        strides::NTuple{N, Int} = strides(parent),
        offset::Int = 0,
        op::F = identity
    ) where {N, F}
    T = Base.promote_op(op, eltype(parent))
    return StridedView{T}(parent, size, strides, offset, op)
end
function StridedView{T}(
        parent::DenseArray,
        size::NTuple{N, Int} = size(parent),
        strides::NTuple{N, Int} = strides(parent),
        offset::Int = 0,
        op::F = identity
    ) where {T, N, F}
    parent′ = _normalizeparent(parent)
    strides′ = _normalizestrides(size, strides)
    return StridedView{T, N, typeof(parent′), F}(parent′, size, strides′, offset, op)
end

StridedView(a::StridedView) = a
StridedView(a::Adjoint) = StridedView(a')'
StridedView(a::Transpose) = transpose(StridedView(transpose(a)))
StridedView(a::Base.SubArray) = sview(StridedView(a.parent), a.indices...)
StridedView(a::Base.ReshapedArray) = sreshape(StridedView(a.parent), a.dims)
function StridedView(a::Base.PermutedDimsArray{T, N, P}) where {T, N, P}
    return permutedims(StridedView(a.parent), P)
end
function StridedView(a::Base.ReinterpretArray{T, N}) where {T, N}
    b = StridedView(a.parent)
    S = eltype(b)
    isbitstype(T) && isbitstype(S) && sizeof(T) == sizeof(S) ||
        throw(ArgumentError("Cannot create StridedView with reinterpretation from $S to $T"))
    b.op isa FN ||
        throw(ArgumentError("Cannot create StridedView with reinterpretation from view with non-identity operation"))
    return StridedView{T}(b.parent, size(b), strides(b), offset(b))
end

# trait
isstrided(a::DenseArray) = true
isstrided(a::StridedView) = true
isstrided(a::Adjoint) = isstrided(a')
isstrided(a::Transpose) = isstrided(transpose(a))
function isstrided(a::Base.SubArray)
    return isstrided(a.parent) && all(Base.Fix2(isa, SliceIndex), a.indices)
end
function isstrided(a::Base.ReshapedArray)
    isstrided(a.parent) || return false
    newsize = a.dims
    oldsize = size(a.parent)
    any(isequal(0), newsize) && return true
    newstrides = _computereshapestrides(
        newsize,
        _simplifydims(oldsize, _strides(a.parent))...
    )
    return !isnothing(newstrides)
end
isstrided(a::Base.PermutedDimsArray) = isstrided(a.parent)
isstrided(a::AbstractArray) = false

# work around annoying Base behavior: it doesn't define strides for complex adjoints
# because of the recursiveness of the definitions, we need to redefine all of them
_strides(a::DenseArray) = strides(a)
_strides(a::Adjoint{<:Any, <:AbstractVector}) = (stride(a.parent, 2), stride(a.parent, 1))
_strides(a::Adjoint{<:Any, <:AbstractMatrix}) = reverse(strides(a.parent))
_strides(a::Transpose{<:Any, <:AbstractVector}) = (stride(a.parent, 2), stride(a.parent, 1))
_strides(a::Transpose{<:Any, <:AbstractMatrix}) = reverse(strides(a.parent))
function _strides(a::PermutedDimsArray{T, N, perm}) where {T, N, perm}
    s = _strides(parent(a))
    return ntuple(d -> s[perm[d]], Val(N))
end
_strides(a::SubArray) = Base.substrides(_strides(a.parent), a.indices)

# Elementary properties
#-----------------------
Base.size(a::StridedView) = a.size
Base.strides(a::StridedView) = a.strides
Base.stride(a::StridedView{<:Any, 0}, n::Int) = 1
function Base.stride(a::StridedView{<:Any, N}, n::Int) where {N}
    return (n <= N) ? a.strides[n] : a.strides[N] * a.size[N]
end
offset(a::StridedView) = a.offset
Base.parent(a::StridedView) = a.parent

# Indexing methods
#------------------
Base.IndexStyle(::Type{<:StridedView}) = Base.IndexCartesian()

# Indexing with N integer arguments
@inline function Base.getindex(a::StridedView{<:Any, N}, I::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(a, I...)
    i = ParentIndex(a.offset + _computeind(I, a.strides))
    @inbounds r = getindex(a, i)
    return r
end
@inline function Base.setindex!(a::StridedView{<:Any, N}, v, I::Vararg{Int, N}) where {N}
    @boundscheck checkbounds(a, I...)
    i = ParentIndex(a.offset + _computeind(I, a.strides))
    @inbounds setindex!(a, v, i)
    return a
end

# Indexing with slice indices to create a new view
@inline function Base.getindex(a::StridedView{T, N}, I::Vararg{SliceIndex, N}) where {T, N}
    return StridedView{T}(
        a.parent,
        _computeviewsize(a.size, I),
        _computeviewstrides(a.strides, I),
        a.offset + _computeviewoffset(a.strides, I),
        a.op
    )
end

# Indexing directly into parent array
struct ParentIndex
    i::Int
end
@propagate_inbounds function Base.getindex(a::StridedView, I::ParentIndex)
    S = Base.promote_op(a.op, eltype(a))
    v = getindex(a.parent, I.i)
    return a.op(v isa S ? v : reinterpret(S, v))
end
@propagate_inbounds function Base.setindex!(a::StridedView, v, I::ParentIndex)
    T = eltype(a)
    v = a.op(convert(T, v))
    S = eltype(a.parent)
    setindex!(a.parent, v isa S ? v : reinterpret(S, v), I.i)
    return a
end

# Specific Base methods that are guaranteed to preserve`StridedView` objects
#----------------------------------------------------------------------------
Base.conj(a::StridedView{<:Real}) = a
function Base.conj(a::StridedView{T}) where {T <: Complex}
    return StridedView{T}(a.parent, a.size, a.strides, a.offset, _conj(a.op))
end
function Base.conj(a::StridedView)
    S = Base.promote_op(a.op, eltype(a))
    newop = _conj(a.op)
    T = Base.promote_op(newop, S)
    return StridedView{T}(a.parent, a.size, a.strides, a.offset, newop)
end

@inline function Base.permutedims(a::StridedView{T, N}, p) where {T, N}
    _isperm(N, p) || throw(ArgumentError("Invalid permutation of length $N: $p"))
    newsize = ntuple(n -> size(a, p[n]), Val(N))
    newstrides = ntuple(n -> stride(a, p[n]), Val(N))
    return StridedView{T}(a.parent, newsize, newstrides, a.offset, a.op)
end

LinearAlgebra.transpose(a::StridedView{<:Number, 2}) = permutedims(a, (2, 1))
LinearAlgebra.adjoint(a::StridedView{<:Number, 2}) = permutedims(conj(a), (2, 1))
function LinearAlgebra.adjoint(a::StridedView{<:Any, 2}) # act recursively, like Base
    S = Base.promote_op(a.op, eltype(a))
    newop = _adjoint(a.op)
    T = Base.promote_op(newop, S)
    return permutedims(StridedView{T}(a.parent, a.size, a.strides, a.offset, newop), (2, 1))
end
function LinearAlgebra.transpose(a::StridedView{<:Any, 2}) # act recursively, like Base
    S = Base.promote_op(a.op, eltype(a))
    newop = _transpose(a.op)
    T = Base.promote_op(newop, S)
    return permutedims(StridedView{T}(a.parent, a.size, a.strides, a.offset, newop), (2, 1))
end

Base.map(::FC, a::StridedView{<:Real}) = a
Base.map(::FT, a::StridedView{<:Number}) = a
Base.map(::FA, a::StridedView{<:Number}) = conj(a)
function Base.map(::FC, a::StridedView)
    T = Base.promote_op(conj, eltype(a))
    return StridedView{T}(a.parent, a.size, a.strides, a.offset, _conj(a.op))
end
function Base.map(::FT, a::StridedView)
    T = Base.promote_op(transpose, eltype(a))
    return StridedView{T}(a.parent, a.size, a.strides, a.offset, _transpose(a.op))
end
function Base.map(::FA, a::StridedView)
    T = Base.promote_op(adjoint, eltype(a))
    return StridedView{T}(a.parent, a.size, a.strides, a.offset, _adjoint(a.op))
end

# Creating or transforming StridedView by slicing
#-------------------------------------------------
# we cannot use Base.view, as this also accepts indices that might not preserve stridedness
sview(a::StridedView{<:Any, N}, I::Vararg{SliceIndex, N}) where {N} = getindex(a, I...)
sview(a::StridedView, I::SliceIndex) = getindex(sreshape(a, (length(a),)), I)

# for StridedView and index arguments which preserve stridedness, we do replace Base.view
# with sview
Base.view(a::StridedView{<:Any, N}, I::Vararg{SliceIndex, N}) where {N} = getindex(a, I...)

# `sview` can be used as a constructor when acting on `AbstractArray` objects
@inline function sview(a::AbstractArray{<:Any, N}, I::Vararg{SliceIndex, N}) where {N}
    return getindex(StridedView(a), I...)
end
@inline function sview(a::AbstractArray, I::SliceIndex)
    return getindex(sreshape(StridedView(a), (length(a),)), I)
end

# Creating or transforming StridedView by reshaping
#---------------------------------------------------
# An error struct for non-strided reshapes
struct ReshapeException{N₁, N₂} <: Exception
    newsize::Dims{N₁}
    oldsize::Dims{N₂}
    strides::Dims{N₂}
end
function Base.show(io::IO, e::ReshapeException)
    msg = "Cannot reshape a StridedView with size $(e.oldsize) and strides $(e.strides) to newsize=$(e.newsize) without allocating, try `sreshape(copy(array), newsize)` or fall back to `reshape(array, newsize)`."
    return print(io, msg)
end

# we cannot use Base.reshape, as this also accepts indices that might not preserve
# stridedness
sreshape(a, args::Vararg{Int}) = sreshape(a, args)
@inline function sreshape(a::StridedView{T}, newsize::Dims) where {T}
    if any(isequal(0), newsize)
        any(isequal(0), size(a)) || throw(DimensionMismatch())
        newstrides = one.(newsize)
    else
        newstrides = _computereshapestrides(newsize, _simplifydims(size(a), strides(a))...)
    end
    isnothing(newstrides) && throw(ReshapeException(newsize, size(a), strides(a)))
    return StridedView{T}(a.parent, newsize, newstrides, a.offset, a.op)
end

sreshape(a::AbstractArray, newsize::Dims) = sreshape(StridedView(a), newsize)

function sreshape(a::LinearAlgebra.AdjointAbsVec, newsize::Dims)
    return sreshape(conj(StridedView(adjoint(a))), newsize)
end
function sreshape(a::LinearAlgebra.TransposeAbsVec, newsize::Dims)
    return sreshape(StridedView(transpose(a)), newsize)
end

# Other methods: `similar`, `copy`
#----------------------------------
function Base.similar(a::StridedView, ::Type{T}, dims::NTuple{N, Int}) where {N, T}
    return StridedView(similar(a.parent, T, dims))
end
Base.copy(a::StridedView) = copyto!(similar(a), a)

# Memory information of a `StridedView`
#---------------------------------------
function Base.unsafe_convert(::Type{Ptr{T}}, a::StridedView{T}) where {T}
    return convert(Ptr{T}, pointer(a.parent, a.offset + 1))
end
function Base.elsize(::Type{<:StridedView{T, N, A}}) where {T, N, A}
    return Base.elsize(A)
end
Base.dataids(a::StridedView) = Base.dataids(a.parent)
