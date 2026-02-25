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
    if !(eltype(C) <: LinearAlgebra.BlasFloat && eltype(A) == eltype(B) == eltype(C))
        return __mul!(C, A, B, α, β)
    end
    # C.op is identity or conj
    if C.op == conj
        if stride(C, 1) < stride(C, 2)
            _mul!(conj(C), conj(A), conj(B), conj(α), conj(β))
        else
            _mul!(C', B', A', conj(α), conj(β))
        end
    elseif stride(C, 1) > stride(C, 2)
        _mul!(transpose(C), transpose(B), transpose(A), α, β)
    else
        _mul!(C, A, B, α, β)
    end
    return C
end

function isblasmatrix(A::StridedView{T, 2}) where {T <: LinearAlgebra.BlasFloat}
    if A.op == identity
        return stride(A, 1) == 1 || stride(A, 2) == 1
    elseif A.op == conj
        return stride(A, 2) == 1
    else # should never happen
        return false
    end
end
function getblasmatrix(A::StridedView{T, 2}) where {T <: LinearAlgebra.BlasFloat}
    if A.op == identity
        if stride(A, 1) == 1
            return A, 'N'
        else
            return transpose(A), 'T'
        end
    else
        return adjoint(A), 'C'
    end
end

# here we will have C.op == :identity && stride(C,1) < stride(C,2)
function _mul!(
        C::StridedView{T, 2}, A::StridedView{T, 2}, B::StridedView{T, 2},
        α::Number, β::Number
    ) where {T <: LinearAlgebra.BlasFloat}
    if stride(C, 1) == 1 && isblasmatrix(A) && isblasmatrix(B)
        nthreads = use_threaded_mul() ? get_num_threads() : 1
        _threaded_blas_mul!(C, A, B, α, β, nthreads)
    else
        return __mul!(C, A, B, α, β)
    end
end

function _threaded_blas_mul!(
        C::StridedView{T, 2}, A::StridedView{T, 2}, B::StridedView{T, 2},
        α::Number, β::Number,
        nthreads
    ) where {T <: LinearAlgebra.BlasFloat}
    m, n = size(C)
    m == size(A, 1) && n == size(B, 2) || throw(DimensionMismatch())
    return if nthreads == 1 || m * n < 1024
        A2, CA = getblasmatrix(A)
        B2, CB = getblasmatrix(B)
        LinearAlgebra.BLAS.gemm!(CA, CB, convert(T, α), A2, B2, convert(T, β), C)
    else
        if m > n
            m2 = round(Int, m / 16) * 8
            nthreads2 = nthreads >> 1
            t = Threads.@spawn _threaded_blas_mul!(
                C[1:($m2), :], A[1:($m2), :], B, α, β,
                $nthreads2
            )
            _threaded_blas_mul!(
                C[(m2 + 1):m, :], A[(m2 + 1):m, :], B, α, β,
                nthreads - nthreads2
            )
            wait(t)
            return C
        else
            n2 = round(Int, n / 16) * 8
            nthreads2 = nthreads >> 1
            t = Threads.@spawn _threaded_blas_mul!(
                C[:, 1:($n2)], A, B[:, 1:($n2)], α, β,
                $nthreads2
            )
            _threaded_blas_mul!(
                C[:, (n2 + 1):n], A, B[:, (n2 + 1):n], α, β,
                nthreads - nthreads2
            )
            wait(t)
            return C
        end
    end
end

# This implementation is faster than LinearAlgebra.generic_matmatmul
function __mul!(
        C::StridedView{<:Any, 2}, A::StridedView{<:Any, 2}, B::StridedView{<:Any, 2},
        α::Number, β::Number
    )
    (size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2) && size(A, 2) == size(B, 1)) ||
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B)), C has size $(size(C))"))

    m, n = size(C)
    k = size(A, 2)
    A2 = sreshape(A, (m, 1, k))
    B2 = sreshape(permutedims(B, (2, 1)), (1, n, k))
    C2 = sreshape(C, (m, n, 1))

    if α == 0 || k == 0
        rmul!(C, β)
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
