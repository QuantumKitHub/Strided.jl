# Shared benchmark cases for the Strided mapreduce machinery.
#
# A `Case` describes one operation that exercises `_mapreducedim!` / the kernel,
# parameterised by ndims `N`, element type `T`, and operation kind `kind`.
# `make_runner(c, sz)` returns a zero-argument closure that performs the op
# in-place on freshly allocated arrays of size `sz`. The same runner is used by
# both the runtime benchmark and the compile/TTFX benchmark, so the two measure
# exactly the same specializations.

using Strided
using Strided: StridedView

@enum OpKind permute add reduce_inner reduce_outer reduce_full

struct Case
    N::Int
    T::DataType
    kind::OpKind
end

name(c::Case) = "$(c.kind)_N$(c.N)_$(c.T)"

# A non-trivial size tuple of N dims with roughly `total` elements, avoiding
# size-1 dims (which would be pushed to the back / fused away).
function sizetuple(N::Int, total::Int)
    d = max(2, round(Int, total^(1 / N)))
    return ntuple(_ -> d, N)
end

function make_runner(c::Case, sz::NTuple{N,Int}) where {N}
    T = c.T
    if c.kind == permute
        p = reverse(ntuple(identity, Val(N)))        # reverse perm: defeats fusion
        src = StridedView(rand(T, sz))
        dst = StridedView(zeros(T, getindex.(Ref(sz), p)))
        return () -> permutedims!(dst, src, p)
    elseif c.kind == add
        a = StridedView(rand(T, sz))
        b = StridedView(rand(T, sz))
        dst = StridedView(zeros(T, sz))
        return () -> map!(+, dst, a, b)
    elseif c.kind == reduce_inner
        A = StridedView(rand(T, sz))
        outsz = ntuple(i -> i == 1 ? 1 : sz[i], Val(N))
        dst = StridedView(zeros(T, outsz))
        return () -> (fill!(dst, zero(T)); Base.mapreducedim!(identity, +, dst, A))
    elseif c.kind == reduce_outer
        A = StridedView(rand(T, sz))
        outsz = ntuple(i -> i == N ? 1 : sz[i], Val(N))
        dst = StridedView(zeros(T, outsz))
        return () -> (fill!(dst, zero(T)); Base.mapreducedim!(identity, +, dst, A))
    elseif c.kind == reduce_full
        A = StridedView(rand(T, sz))
        return () -> sum(A)
    else
        error("unknown kind $(c.kind)")
    end
end

# Build the full case grid.
function all_cases(; Ns, Ts, kinds)
    cs = Case[]
    for kind in kinds, T in Ts, N in Ns
        push!(cs, Case(N, T, kind))
    end
    return cs
end
