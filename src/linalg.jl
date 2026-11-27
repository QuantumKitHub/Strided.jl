# linear algebra
LinearAlgebra.rmul!(dst::StridedView, α::Number) = mul!(dst, dst, α)
LinearAlgebra.lmul!(α::Number, dst::StridedView) = mul!(dst, α, dst)

function LinearAlgebra.mul!(
        dst::StridedView{<:Number, N}, α::Number,
        src::StridedView{<:Number, N}
    ) where {N}
    if α == 1
        copy!(dst, src)
    else
        dst .= α .* src
    end
    return dst
end
function LinearAlgebra.mul!(
        dst::StridedView{<:Number, N}, src::StridedView{<:Number, N},
        α::Number
    ) where {N}
    if α == 1
        copy!(dst, src)
    else
        dst .= src .* α
    end
    return dst
end
function LinearAlgebra.axpy!(
        a::Number, X::StridedView{<:Number, N},
        Y::StridedView{<:Number, N}
    ) where {N}
    if a == 1
        Y .= X .+ Y
    else
        Y .= a .* X .+ Y
    end
    return Y
end
function LinearAlgebra.axpby!(
        a::Number, X::StridedView{<:Number, N},
        b::Number, Y::StridedView{<:Number, N}
    ) where {N}
    if b == 1
        axpy!(a, X, Y)
    elseif b == 0
        mul!(Y, a, X)
    else
        Y .= a .* X .+ b .* Y
    end
    return Y
end

function LinearAlgebra.mul!(
        C::StridedView{T, 2},
        A::StridedView{<:Any, 2}, B::StridedView{<:Any, 2},
        α::Number = true, β::Number = false
    ) where {T}
    # an empty product only scales `C`, which does not need BLAS; GPU BLAS libraries can not
    # handle the empty arrays anyway
    if iszero(size(A, 2))
        (size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2) && iszero(size(B, 1))) ||
            _throw_mul_dimensionmismatch(size(A), size(B), size(C))
        return _scale!(C, β)
    end
    if !(eltype(C) <: LinearAlgebra.BlasFloat && eltype(A) == eltype(B) == eltype(C))
        return __mul!(C, A, B, α, β)
    end
    # C.op is identity or conj
    if C.op == conj
        if !_transposeoutput(C)
            _mul!(conj(C), conj(A), conj(B), conj(α), conj(β))
        else
            _mul!(C', B', A', conj(α), conj(β))
        end
    elseif _transposeoutput(C)
        _mul!(transpose(C), transpose(B), transpose(A), α, β)
    else
        _mul!(C, A, B, α, β)
    end
    return C
end

# Whether to compute the transpose of `C`: BLAS can only write to `C` directly if it has a BLAS
# layout (see `_isblaslayout`), so prefer the orientation that has one, and otherwise the one
# with the smaller stride along the rows.
function _transposeoutput(C::StridedView{<:Any, 2})
    (m, n), (s1, s2) = size(C), strides(C)
    _isblaslayout(m, n, s1, s2) && return false
    return _isblaslayout(n, m, s2, s1) || s1 > s2
end

# `C = β * C`, where `β == 0` overwrites `C`, which may contain `NaN`s, e.g. if it is uninitialized
_scale!(C::StridedView, β::Number) = iszero(β) ? fill!(C, zero(eltype(C))) : rmul!(C, β)

# Whether an `m × n` matrix with strides `(s1, s2)` can be passed to BLAS as a column-major
# matrix: unit stride along the rows, and a leading dimension (stride along the columns) of at
# least the number of rows, which also excludes negative strides and overlapping columns.
# The stride of a dimension of size 1 does not matter, and neither do the strides of an empty
# matrix: `_blaslayout` replaces them by values that satisfy these conditions.
function _isblaslayout(m::Int, n::Int, s1::Int, s2::Int)
    (m == 0 || n == 0) && return true
    return (m == 1 || s1 == 1) && (n == 1 || s2 >= m)
end
function _blaslayout(A::StridedView{<:Any, 2})
    (m, n), (s1, s2) = size(A), strides(A)
    isempty = m == 0 || n == 0
    s1′ = ifelse(isempty | (m == 1), 1, s1)
    s2′ = ifelse(isempty | (n == 1), max(m, 1), s2)
    return unsafe_stridedview(A, (m, n), (s1′, s2′))
end

function isblasmatrix(A::StridedView{T, 2}) where {T <: LinearAlgebra.BlasFloat}
    (m, n), (s1, s2) = size(A), strides(A)
    if A.op == identity
        return _isblaslayout(m, n, s1, s2) || _isblaslayout(n, m, s2, s1)
    elseif A.op == conj
        return _isblaslayout(n, m, s2, s1)
    else # should never happen
        return false
    end
end
getblasmatrix(A::StridedView{T, 2}) where {T <: LinearAlgebra.BlasFloat} = _blasview(A)
# Returns a matrix with a BLAS layout (see `_isblaslayout`), together with the BLAS operation
# that has to be applied to it to obtain `A`. `A` has to satisfy `isblasmatrix(A)`.
function _blasview(A::StridedView{T, 2}) where {T <: LinearAlgebra.BlasFloat}
    if A.op == identity
        if _isblaslayout(size(A)..., strides(A)...)
            return _blaslayout(A), 'N'
        else
            return _blaslayout(transpose(A)), 'T'
        end
    else
        return _blaslayout(adjoint(A)), 'C'
    end
end

# here we will have C.op == identity, and C has a BLAS layout if possible (see `_transposeoutput`)
function _mul!(
        C::StridedView{T, 2}, A::StridedView{T, 2}, B::StridedView{T, 2},
        α::Number, β::Number
    ) where {T <: LinearAlgebra.BlasFloat}
    if isblasmatrix(A) && isblasmatrix(B) && isblasmatrix(C)
        return blas_mul!(C, A, B, α, β)
    else
        return __mul!(C, A, B, α, β)
    end
end

# for CPU based arrays, this is valid
function blas_mul!(
        C::StridedView{T, 2}, A::StridedView{T, 2}, B::StridedView{T, 2},
        α::Number, β::Number
    ) where {T <: LinearAlgebra.BlasFloat}
    m, n = size(C)
    m == size(A, 1) && n == size(B, 2) || _throw_dimensionmismatch()
    nthreads = use_threaded_mul() ? get_num_threads() : 1
    A2, CA = getblasmatrix(A)
    B2, CB = getblasmatrix(B)
    C2, _ = getblasmatrix(C) # `_mul!` guarantees `C.op == identity` and a BLAS layout
    _threaded_blas_mul!(C2, A2, CA, B2, CB, convert(T, α), convert(T, β), nthreads)
    return C
end

_split_row(A, m) = (A[1:m, :], A[(m + 1):end, :])
_split_col(A, m) = (A[:, 1:m], A[:, (m + 1):end])

function _threaded_blas_mul!(
        C::StridedView{T, 2},
        A::StridedView{T, 2}, CA,
        B::StridedView{T, 2}, CB,
        α::Number, β::Number,
        nthreads
    ) where {T <: LinearAlgebra.BlasFloat}
    m, n = size(C)
    if nthreads == 1 || m * n < 1024
        LinearAlgebra.BLAS.gemm!(CA, CB, α, A, B, β, C)
        return C
    end

    if m > n
        m2 = round(Int, m / 16) * 8
        Chead, Ctail = _split_row(C, m2)
        Ahead, Atail = CA == 'N' ? _split_row(A, m2) : _split_col(A, m2)
        Bhead = Btail = B
    else
        n2 = round(Int, n / 16) * 8
        Chead, Ctail = _split_col(C, n2)
        Ahead = Atail = A
        Bhead, Btail = CB == 'N' ? _split_col(B, n2) : _split_row(B, n2)
    end
    nthreads1 = nthreads >> 1
    nthreads2 = nthreads - nthreads1
    t = Threads.@spawn _threaded_blas_mul!($Chead, $Ahead, CA, $Bhead, CB, α, β, nthreads1)
    _threaded_blas_mul!(Ctail, Atail, CA, Btail, CB, α, β, nthreads2)
    wait(t)
    return C
end

# This implementation is faster than LinearAlgebra.generic_matmatmul
function __mul!(
        C::StridedView{<:Any, 2}, A::StridedView{<:Any, 2}, B::StridedView{<:Any, 2},
        α::Number, β::Number
    )
    (size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2) && size(A, 2) == size(B, 1)) ||
        _throw_mul_dimensionmismatch(size(A), size(B), size(C))

    m, n = size(C)
    k = size(A, 2)
    A2 = sreshape(A, (m, 1, k))
    B2 = sreshape(permutedims(B, (2, 1)), (1, n, k))
    C2 = sreshape(C, (m, n, 1))

    if α == 0 || k == 0
        _scale!(C, β)
    elseif α == 1
        if β == 0
            _mapreducedim!(*, +, zero, (m, n, k), (C2, A2, B2))
        elseif β == 1
            _mapreducedim!(*, +, nothing, (m, n, k), (C2, A2, B2))
        else
            _mapreducedim!(*, +, x -> x * β, (m, n, k), (C2, A2, B2))
        end
    else
        f = (x, y) -> (x * y * α)
        if β == 0
            _mapreducedim!(f, +, zero, (m, n, k), (C2, A2, B2))
        elseif β == 1
            _mapreducedim!(f, +, nothing, (m, n, k), (C2, A2, B2))
        else
            _mapreducedim!(f, +, x -> x * β, (m, n, k), (C2, A2, B2))
        end
    end
    return C
end
