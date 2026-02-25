function Base.Array(a::StridedView)
    b = Array{eltype(a)}(undef, size(a))
    copy!(StridedView(b), a)
    return b
end

function (Base.Array{T})(a::StridedView{S, N}) where {T, S, N}
    b = Array{T}(undef, size(a))
    copy!(StridedView(b), a)
    return b
end

function (Base.Array{T, N})(a::StridedView{S, N}) where {T, S, N}
    b = Array{T}(undef, size(a))
    copy!(StridedView(b), a)
    return b
end
