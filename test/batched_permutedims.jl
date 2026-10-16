# Independent reference/oracle and adversarial fixtures for
# Strided.batched_permutedims!/plan_batched_permutedims. This file is self-contained (run
# via `include`), does not depend on runtests.jl, and must never consult planner
# internals: only `perm` (Base's `permutedims` convention) and plain array indexing are
# used to build the oracle.
#
# Note on scope: this planner does not check that inputs are densely packed, and does not
# check aliasing at all (see the module docstring on `plan_batched_permutedims`). So
# non-dense and negative-stride views are exercised below as *correctness* fixtures (they
# are genuinely, correctly supported, not merely tolerated), while aliasing/overlap and
# malformed-view cases are deliberately NOT exercised here as error-throwing tests, since
# they no longer throw and actually running them would either produce an unspecified
# (silently wrong) result or, for a view that doesn't fit inside its own parent, write out
# of bounds -- there is nothing safe to assert about either outcome.

using Test
using Random
using LinearAlgebra
using Strided
using Strided: StridedView

Random.seed!(20240915)

# ----------------------------------------------------------------------------
# Reference oracle: two independent computations of the same
# thing, cross-checked against each other and never against planner code.
# ----------------------------------------------------------------------------

# explicit loop
function ref_loop(src::AbstractArray, perm)
    N0 = length(perm)
    out = Array{eltype(src)}(undef, ntuple(j -> size(src, perm[j]), N0))
    for I in CartesianIndices(size(src))
        out[ntuple(j -> I[perm[j]], N0)...] = src[I]
    end
    return out
end

# second, independent cross-check via Base.permutedims
function ref_base(src::AbstractArray, perm)
    return permutedims(Array(src), perm)
end

# D11: bit-preserving comparison, never isapprox. Reuses only the `compare`
# *structure* of test/gpu.jl (not its isapprox tolerance semantics).
refequal(a::AbstractArray, b::AbstractArray) = isequal(Array(a), Array(b))

# builds the oracle output for one batch and self-checks the two independent
# routes agree, so a bug in `ref_loop` cannot silently pass as ground truth
function refbatch(srcs, perm)
    refs = map(s -> ref_loop(s, perm), srcs)
    for (s, r) in zip(srcs, refs)
        @test refequal(ref_base(s, perm), r)
    end
    return refs
end

# full round-trip harness for a *valid* fixture: compute the oracle, run the
# public API, check identity of the return value, check dst == oracle, and
# check srcs were not mutated (read-only sources, D5/section 9).
function runcase!(dsts, srcs, perm)
    srcsnap = map(deepcopy, srcs)
    refs = refbatch(srcs, perm)
    out = Strided.batched_permutedims!(dsts, srcs, perm)
    @test out === dsts
    for (d, r) in zip(dsts, refs)
        @test refequal(d, r)
    end
    for (s, s0) in zip(srcs, srcsnap)
        @test refequal(s, s0)
    end
    return nothing
end

@testset "Strided.batched_permutedims! (reference oracle self-check)" begin
    # sanity: the oracle itself is correct before it judges anyone else's code.
    # (also duplicated standalone in the scratchpad, run against plain Base.)
    A = rand(3, 4, 5)
    p = (3, 1, 2)
    @test refequal(ref_loop(A, p), permutedims(A, p))
    @test refequal(ref_loop(A, p), ref_base(A, p))
    B = rand(ComplexF64, 2, 0, 4) # empty array, still a valid oracle input
    @test refequal(ref_loop(B, (2, 3, 1)), permutedims(B, (2, 3, 1)))
end

@testset "Strided.batched_permutedims! (correctness fixtures)" begin
    @testset "empty batch (B == 0)" begin
        dsts = Matrix{Float64}[]
        srcs = Matrix{Float64}[]
        out = Strided.batched_permutedims!(dsts, srcs, (2, 1))
        @test out === dsts
    end

    @testset "all tensors empty (totaltiles == 0)" begin
        srcs = [zeros(3, 0, 4), zeros(0, 2, 4), zeros(3, 2, 0)]
        dsts = [zeros(size(s, 2), size(s, 1), size(s, 3)) for s in srcs]
        runcase!(dsts, srcs, (2, 1, 3))
    end

    @testset "empty tensors interleaved (repeated prefix values)" begin
        srcs = Any[rand(3, 4), zeros(0, 4), zeros(3, 0), rand(2, 5), zeros(0, 0)]
        perm = (2, 1)
        dsts = Any[zeros(size(s, 2), size(s, 1)) for s in srcs]
        runcase!(dsts, srcs, perm)
    end

    @testset "rank 0 (perm == ())" begin
        srcs = [fill(1.5), fill(-2.0), fill(NaN)]
        dsts = [fill(0.0), fill(0.0), fill(0.0)]
        runcase!(dsts, srcs, ())
    end

    @testset "singleton axis: present in some tensors only (must not be dropped)" begin
        s1 = rand(3, 1, 4) # axis 2 singleton
        s2 = rand(3, 2, 4) # axis 2 not singleton
        perm = (3, 1, 2)
        srcs = [s1, s2]
        dsts = [zeros(ntuple(j -> size(s, perm[j]), 3)) for s in srcs]
        runcase!(dsts, srcs, perm)
    end

    @testset "singleton axis: present in every tensor (dropped internally)" begin
        s1 = rand(3, 1, 4)
        s2 = rand(5, 1, 4)
        perm = (3, 1, 2)
        srcs = [s1, s2]
        dsts = [zeros(ntuple(j -> size(s, perm[j]), 3)) for s in srcs]
        runcase!(dsts, srcs, perm)
    end

    @testset "singleton axis with a fabricated stride still addresses correctly" begin
        # StridedViews rewrites a singleton axis's stride to whatever is convenient
        # internally; here permuting turns axis 2 (originally stride 4) into a singleton
        # axis with a fabricated stride of 16. Since that axis's only coordinate is 0,
        # its stride value (fabricated or not) never actually gets multiplied by
        # anything nonzero, so this is harmless -- demonstrated directly rather than
        # asserted as an invariant.
        A = rand(4, 1, 4) # strides(StridedView(A)) == (1, 4, 4)
        sv = StridedView(A)
        wv = permutedims(sv, (3, 2, 1))
        @test strides(wv) == (4, 16, 1)
        dst = zeros(size(wv))
        runcase!([dst], [wv], (1, 2, 3))
    end

    @testset "perm given as AbstractVector{<:Integer}, not NTuple" begin
        s = rand(2, 3, 4)
        perm = [2, 3, 1]
        dst = zeros(3, 4, 2)
        runcase!([dst], [s], perm)
    end

    @testset "real transpose/adjoint normalize to identity op (accepted)" begin
        A = rand(4, 5)
        srcT = transpose(A)
        dst = zeros(5, 4)
        runcase!([dst], [srcT], (1, 2)) # identity perm on the transposed view
        B = rand(4, 5)
        srcA = adjoint(B) # real adjoint == transpose, still identity op
        dst2 = zeros(5, 4)
        runcase!([dst2], [srcA], (1, 2))
    end

    @testset "nonzero offsets, disjoint views of one parent" begin
        parent = collect(1.0:100.0)
        src = reshape(view(parent, 1:24), 4, 6)
        dst = reshape(view(parent, 25:48), 6, 4)
        runcase!([dst], [src], (2, 1))
    end

    @testset "srcs[b] === srcs[b'] accepted (source/source overlap allowed)" begin
        s = rand(3, 4)
        srcs = [s, s]
        dsts = [zeros(4, 3), zeros(4, 3)]
        runcase!(dsts, srcs, (2, 1))
    end

    @testset "order-inference heuristic falls back cleanly when no tensor is unambiguous" begin
        # Every tensor here has a singleton axis, so `_infer_order` never finds a
        # fully-unambiguous tensor to copy the order from and falls back to the plain
        # 1:N0 axis order. Correctness must not depend on that choice: addressing always
        # uses each tensor's own real strides, regardless of which heuristic order the
        # planner picked for scheduling.
        A = permutedims(StridedView(rand(5, 3, 1)), (1, 3, 2)) # size (5,1,3), dense
        B = permutedims(StridedView(rand(4, 1, 3)), (2, 1, 3)) # size (1,4,3), dense
        perm = (1, 2, 3)
        srcs = [A, B]
        dsts = [zeros(size(s)) for s in srcs]
        runcase!(dsts, srcs, perm)
    end

    @testset "non-dense (strided) view: correctly supported, not merely tolerated" begin
        s = view(rand(10), 1:2:9) # stride 2, not unit-stride
        dst = zeros(5)
        runcase!([dst], [s], (1,))
    end

    @testset "negative-stride view: correctly supported" begin
        A = rand(6)
        s = @view A[6:-1:1]
        dst = zeros(6)
        runcase!([dst], [s], (1,))
    end

    @testset "dominant tensor plus many tiny ones, spread of sizes" begin
        srcs = Any[rand(97, 131)]
        for n in (1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65)
            push!(srcs, rand(n, n + 1))
        end
        perm = (2, 1)
        dsts = Any[zeros(size(s, 2), size(s, 1)) for s in srcs]
        runcase!(dsts, srcs, perm)
    end

    @testset "tile-edge spread, rank 3, transpose-family-shaped batch" begin
        srcs = Any[]
        for a in (1, 2, 8, 9, 63, 64, 65), b in (1, 3, 16, 17)
            push!(srcs, rand(a, b, 2))
        end
        perm = (3, 1, 2)
        dsts = Any[zeros(ntuple(j -> size(s, perm[j]), 3)) for s in srcs]
        runcase!(dsts, srcs, perm)
    end
end

@testset "Strided.batched_permutedims! (plan reuse)" begin
    s1 = rand(6, 7)
    d1 = zeros(7, 6)
    perm = (2, 1)
    plan = Strided.plan_batched_permutedims([d1], [s1], perm)
    refs = refbatch([s1], perm)
    dsts1 = [d1]
    out = Strided.batched_permutedims!(dsts1, [s1], plan)
    @test out === dsts1
    @test refequal(d1, refs[1])
    dsts1b = [d1]
    out2 = Strided.batched_permutedims!(dsts1b, [s1], perm)
    @test out2 === dsts1b
    @test refequal(d1, refs[1])

    # plan reused with a *different* but shape/stride-compatible array pair
    s2 = rand(6, 7)
    d2 = zeros(7, 6)
    refs2 = refbatch([s2], perm)
    Strided.batched_permutedims!([d2], [s2], plan)
    @test refequal(d2, refs2[1])
end

@testset "Strided.batched_permutedims! (validation: DimensionMismatch)" begin
    s = rand(3, 4)
    d = zeros(4, 3)

    @testset "length(dsts) != length(srcs)" begin
        @test_throws DimensionMismatch Strided.batched_permutedims!([d, d], [s], (2, 1))
    end

    @testset "ndims != N0" begin
        s3 = rand(3, 4, 1)
        @test_throws DimensionMismatch Strided.batched_permutedims!([d], [s3], (2, 1))
    end

    @testset "size(dst)[j] != size(src)[perm[j]]" begin
        badd = zeros(3, 4) # should be (4,3) for perm (2,1)
        @test_throws DimensionMismatch Strided.batched_permutedims!([badd], [s], (2, 1))
    end
end

@testset "Strided.batched_permutedims! (validation: ArgumentError)" begin
    @testset "invalid perm (not isperm)" begin
        s = rand(3, 4)
        d = zeros(4, 3)
        @test_throws ArgumentError Strided.batched_permutedims!([d], [s], (1, 1))
    end

    @testset "mixed element type within one side (non-concrete eltype)" begin
        s1 = rand(Float64, 3, 4)
        s2 = rand(Float32, 3, 4)
        d1 = zeros(4, 3)
        d2 = zeros(4, 3)
        @test_throws ArgumentError Strided.batched_permutedims!([d1, d2], Any[s1, s2], (2, 1))
    end

    @testset "matching eltype within each side, mismatched across sides" begin
        s = rand(Float64, 3, 4)
        d = zeros(Float32, 4, 3)
        @test_throws ArgumentError Strided.batched_permutedims!([d], [s], (2, 1))
    end

    @testset "non-identity op: adjoint/conj of complex array rejected" begin
        A = rand(ComplexF64, 4, 5)
        d = zeros(ComplexF64, 5, 4)
        @test_throws ArgumentError Strided.batched_permutedims!([d], [adjoint(A)], (1, 2))
        @test_throws ArgumentError Strided.batched_permutedims!([d], Any[conj(StridedView(A))], (1, 2))
    end

    # Aliasing (dsts[b] === srcs[b], overlapping destinations, overlapping src/dst into
    # one parent), non-dense views, negative strides, and a view that doesn't fit inside
    # its own parent are all UNCHECKED here on purpose (see the file header and
    # `plan_batched_permutedims`'s docstring). Non-dense/negative-stride views are
    # exercised as *correctness* fixtures above, not here; aliasing and malformed-parent
    # cases are not exercised at all, since there is nothing safe to assert about their
    # outcome once the checks are gone (see also: a genuinely cyclic/incompatible common
    # order between two tensors used to be rejected here too, but order inference is now
    # only a scheduling heuristic that falls back silently -- see "order-inference
    # heuristic falls back cleanly..." above, which exercises exactly that fixture as a
    # correctness case instead of an error case).

    @testset "Tuple collections get no method" begin
        s = rand(3, 4)
        d = zeros(4, 3)
        @test_throws Union{MethodError, UndefVarError} Strided.batched_permutedims!((d,), (s,), (2, 1))
    end
end

@testset "Strided.batched_permutedims! (metadata-only OverflowError, no allocation)" begin
    # Real backing parents are tiny (1 element); declared size/strides deliberately
    # overflow Int64 in a way that wraps to a value <= the (tiny) real parent length,
    # so V11 containment passes on the wrapped product and the genuine overflow is
    # only caught later by V13's checked_* arithmetic. No large array is allocated.
    n1 = 2^62
    n2 = 4 # n1 * n2 == 2^64, wraps to 0 in Int64 arithmetic
    srcparent = zeros(1)
    dstparent = zeros(1)
    src = StridedView(srcparent, (n1, n2), (1, n1), 0)
    dst = StridedView(dstparent, (n1, n2), (1, n1), 0)
    @test_throws OverflowError Strided.plan_batched_permutedims([dst], [src], (1, 2))
end

@testset "Strided.plan_batched_permutedims (strategy keyword, CPU-backed batches)" begin
    @testset "default strategy resolves to BP_AUTO" begin
        s = rand(3, 4)
        d = zeros(4, 3)
        plan = Strided.plan_batched_permutedims([d], [s], (2, 1))
        @test plan.strategy == Strided.BP_AUTO
    end

    @testset "explicit BP_AUTO on CPU arrays succeeds and resolves to BP_AUTO" begin
        s = rand(3, 4)
        d = zeros(4, 3)
        plan = Strided.plan_batched_permutedims([d], [s], (2, 1); strategy = Strided.BP_AUTO)
        @test plan.strategy == Strided.BP_AUTO
        refs = refbatch([s], (2, 1))
        Strided.batched_permutedims!([d], [s], plan)
        @test refequal(d, refs[1])
    end

    @testset "non-AUTO strategy on CPU arrays is rejected before any write" begin
        for strategy in (Strided.BP_ELEMENTWISE, Strided.BP_THREADTILE, Strided.BP_GROUPTILE)
            s = rand(3, 4)
            d = zeros(4, 3)
            @test_throws ArgumentError Strided.plan_batched_permutedims([d], [s], (2, 1); strategy)
            @test_throws ArgumentError Strided.batched_permutedims!([d], [s], (2, 1); strategy)
            @test all(iszero, d) # rejected before any write reached dst
        end
    end

    @testset "plan reuse preserves the resolved strategy unchanged" begin
        s1 = rand(6, 7)
        d1 = zeros(7, 6)
        perm = (2, 1)
        plan = Strided.plan_batched_permutedims([d1], [s1], perm; strategy = Strided.BP_AUTO)
        @test plan.strategy == Strided.BP_AUTO

        # rebuild directly against a different-but-compatible array pair, and check the
        # rebuilt plan's strategy field matches the original plan's, unchanged
        s2 = rand(6, 7)
        d2 = zeros(7, 6)
        dviews = map(StridedView, [d2])
        sviews = map(StridedView, [s2])
        newplan = Strided._rebuild_descriptors(dviews, sviews, plan)
        @test newplan.strategy == plan.strategy == Strided.BP_AUTO

        # and via the public reuse path, executing still works and the plan is unaffected
        refs2 = refbatch([s2], perm)
        Strided.batched_permutedims!([d2], [s2], plan)
        @test refequal(d2, refs2[1])
        @test plan.strategy == Strided.BP_AUTO
    end
end

@testset "Strided._rebuild_descriptors (plan reuse: same-descriptor fast path)" begin
    perm = (3, 1, 2)
    N0 = length(perm)
    srcs = [rand(4, 5, 6), rand(2, 3, 7), rand(1, 8, 2), rand(6, 6, 6)]
    dsts = [zeros(ntuple(j -> size(s, perm[j]), N0)) for s in srcs]
    plan = Strided.plan_batched_permutedims(dsts, srcs, perm)
    # a uniformly-typed view vector, so one entry can be swapped for a view with a
    # different offset/strides without changing the vector's element type
    sviews0 = map(StridedView, srcs)
    dviews0 = map(StridedView, dsts)

    @testset "same arrays again: the identical plan object comes back" begin
        dviews, sviews, _ = Strided._normalize_and_check(dsts, srcs, N0)
        same = Strided._rebuild_descriptors(dviews, sviews, plan)
        @test same === plan
        @test same.descs === plan.descs && same.cpublocks === plan.cpublocks
        # and it agrees with a freshly built reference plan, field for field
        fresh = Strided.plan_batched_permutedims(dsts, srcs, perm)
        @test same.descs == fresh.descs && same.cpublocks == fresh.cpublocks
        @test same.prefix == fresh.prefix && same.totaltiles == fresh.totaltiles
        # fully inferred and allocation-free on this path
        @test (@inferred Strided._rebuild_descriptors(dviews, sviews, plan)) === plan
        rebuild(dv, sv, p) = Strided._rebuild_descriptors(dv, sv, p)
        rebuild(dviews, sviews, plan)
        @test (@allocated rebuild(dviews, sviews, plan)) == 0
    end

    @testset "one source swapped for a view with a different offset: new plan, prefix copied" begin
        big = rand(20, 3, 7)
        sviews2 = copy(sviews0)
        sviews2[2] = view(StridedView(big), 9:10, :, :) # same shape as srcs[2]; offset 8, strides (1,20,60)
        @test size(sviews2[2]) == size(srcs[2]) && Strided.offset(sviews2[2]) != 0
        changed = Strided._rebuild_descriptors(dviews0, sviews2, plan)
        @test changed !== plan
        @test changed.descs !== plan.descs && changed.cpublocks !== plan.cpublocks
        @test changed.descs[2] != plan.descs[2]
        @test changed.descs[[1, 3, 4]] == plan.descs[[1, 3, 4]]
        @test changed.cpublocks[[1, 3, 4]] == plan.cpublocks[[1, 3, 4]]
        ref2 = Strided.plan_batched_permutedims(dviews0, sviews2, perm)
        @test changed.descs == ref2.descs && changed.cpublocks == ref2.cpublocks
        # the original plan is untouched
        @test plan.descs == Strided.plan_batched_permutedims(dsts, srcs, perm).descs
        # and the public reuse path with the swapped arrays computes the right thing
        refs = refbatch(sviews2, perm)
        Strided.batched_permutedims!(dviews0, sviews2, plan)
        for (d, r) in zip(dviews0, refs)
            @test refequal(d, r)
        end
    end

    @testset "first tensor differs: prefix copy is a no-op, nothing before it to reuse" begin
        big = rand(20, 5, 6)
        sviews6 = copy(sviews0)
        sviews6[1] = view(StridedView(big), 9:12, :, :) # same shape as srcs[1] (4,5,6), different offset/strides
        changed = Strided._rebuild_descriptors(dviews0, sviews6, plan)
        @test changed !== plan
        @test changed.descs[1] != plan.descs[1]
        @test changed.descs[[2, 3, 4]] == plan.descs[[2, 3, 4]]
        @test changed.cpublocks[[2, 3, 4]] == plan.cpublocks[[2, 3, 4]]
        ref6 = Strided.plan_batched_permutedims(dviews0, sviews6, perm)
        @test changed.descs == ref6.descs && changed.cpublocks == ref6.cpublocks
        @test plan.descs == Strided.plan_batched_permutedims(dsts, srcs, perm).descs
    end

    @testset "last tensor differs: the whole prefix is copied, nothing after it to write" begin
        big = rand(20, 6, 6)
        sviews7 = copy(sviews0)
        sviews7[4] = view(StridedView(big), 9:14, :, :) # same shape as srcs[4] (6,6,6), different offset/strides
        changed = Strided._rebuild_descriptors(dviews0, sviews7, plan)
        @test changed !== plan
        @test changed.descs[4] != plan.descs[4]
        @test changed.descs[[1, 2, 3]] == plan.descs[[1, 2, 3]]
        @test changed.cpublocks[[1, 2, 3]] == plan.cpublocks[[1, 2, 3]]
        ref7 = Strided.plan_batched_permutedims(dviews0, sviews7, perm)
        @test changed.descs == ref7.descs && changed.cpublocks == ref7.cpublocks
        @test plan.descs == Strided.plan_batched_permutedims(dsts, srcs, perm).descs
    end

    @testset "one source swapped for a fresh same-shape Array: descriptors equal, same plan" begin
        # A different array object with identical shape, strides and offset yields a
        # bitwise-identical descriptor, so the plan object is reused; that carries no
        # stale-array risk because the executor is handed the new views separately.
        srcs3 = copy(srcs)
        srcs3[3] = rand(size(srcs[3])...)
        dviews3, sviews3, _ = Strided._normalize_and_check(dsts, srcs3, N0)
        @test Strided._rebuild_descriptors(dviews3, sviews3, plan) === plan
        refs = refbatch(srcs3, perm)
        Strided.batched_permutedims!(dsts, srcs3, plan)
        for (d, r) in zip(dsts, refs)
            @test refequal(d, r)
        end
    end

    @testset "shape mismatch still throws, before and after the first differing tensor" begin
        # mismatch on the very first tensor (nothing has been rebuilt yet)
        srcs4 = copy(srcs)
        srcs4[1] = rand(4, 5, 7)
        dsts4 = copy(dsts)
        dsts4[1] = zeros(7, 4, 5)
        dviews4, sviews4, _ = Strided._normalize_and_check(dsts4, srcs4, N0)
        @test_throws ArgumentError Strided._rebuild_descriptors(dviews4, sviews4, plan)
        # mismatch on a later tensor, after an earlier tensor's descriptor already differed
        big = rand(20, 3, 7)
        sviews5 = copy(sviews0)
        sviews5[2] = view(StridedView(big), 9:10, :, :)
        sviews5[4] = StridedView(rand(6, 6, 5))
        dviews5 = copy(dviews0)
        dviews5[4] = StridedView(zeros(5, 6, 6))
        @test_throws ArgumentError Strided._rebuild_descriptors(dviews5, sviews5, plan)
        @test plan.descs == Strided.plan_batched_permutedims(dsts, srcs, perm).descs
    end

    @testset "_normview infers a concrete view vector for concretely-typed inputs" begin
        v = @inferred Strided._normview(srcs)
        @test isconcretetype(eltype(v)) && v == map(StridedView, srcs)
        @inferred Strided._normalize_and_check(dsts, srcs, N0)
        # `Any[...]` inputs still narrow to the actual common view type, or get rejected
        anyv = Strided._normview(Any[rand(3, 4), zeros(0, 4)])
        @test isconcretetype(eltype(anyv))
        @test !isconcretetype(eltype(Strided._normview(Any[rand(Float64, 3, 4), rand(Float32, 3, 4)])))
        e = Strided._normview(Matrix{Float64}[])
        @test isempty(e) && isconcretetype(eltype(e))
    end
end

# ----------------------------------------------------------------------------
# Scaled path: dsts[b] = alpha[b] * permutedims(srcs[b], perm) + beta[b] * dsts[b].
# Unlike the plain copy, this computes, so `isequal` is only a sound oracle on data whose
# every intermediate is exactly representable: small integer-valued entries and small
# integer / half-integer coefficients, for which IEEE arithmetic gives bit-identical results
# whatever the evaluation order or multiply-add fusion. A mismatch here is a real finding,
# never something to relax to `isapprox`.
# ----------------------------------------------------------------------------

exactdata(::Type{T}, dims) where {T <: Real} = T.(rand(-8:8, dims))
exactdata(::Type{T}, dims) where {T <: Complex} = T.(complex.(rand(-8:8, dims), rand(-8:8, dims)))
exactcoeffs(::Type{T}) where {T <: Real} = T[-3, -1, 0, 0.5, 1, 2, 5]
exactcoeffs(::Type{T}) where {T <: Complex} = T[-3, -1, 0, 0.5, 1, 2, 5, 1 - 2im]

# reference in plain Array arithmetic in T; `beta == 0` must never read the destination
scaledref(a, s, b, d0, perm) = iszero(b) ? a .* ref_base(s, perm) : a .* ref_base(s, perm) .+ b .* d0

# alpha/beta cycling through the exact set with a shift, so tensor 1 gets beta == 0, tensor 3
# gets alpha == 0, and every later tensor gets some nonzero beta
function scaledcoeffs(::Type{T}, B::Int) where {T}
    cs = exactcoeffs(T)
    n = length(cs)
    return [cs[mod1(b, n)] for b in 1:B], [cs[mod1(b + 2, n)] for b in 1:B]
end

dstshape(s, perm) = ntuple(j -> size(s, perm[j]), length(perm))

# destinations: NaN-poisoned wherever beta == 0 (they must come out clean), exact data elsewhere
function scaleddsts(::Type{T}, srcs, perm, beta) where {T}
    return [iszero(beta[b]) ? fill(T(NaN), dstshape(srcs[b], perm)) : exactdata(T, dstshape(srcs[b], perm))
            for b in eachindex(srcs)]
end

function scaledcase!(dsts, srcs, perm, alpha, beta; plan = nothing)
    T = eltype(dsts[1])
    d0 = map(copy, dsts)
    srcsnap = map(deepcopy, srcs)
    out = plan === nothing ? Strided.batched_permutedims!(dsts, srcs, perm; alpha, beta) :
        Strided.batched_permutedims!(dsts, srcs, plan; alpha, beta)
    @test out === dsts
    for b in eachindex(dsts)
        @test refequal(dsts[b], scaledref(T(alpha[b]), srcs[b], T(beta[b]), d0[b], perm))
    end
    for (s, s0) in zip(srcs, srcsnap)
        @test refequal(s, s0)
    end
    return dsts
end

@testset "Strided.batched_permutedims! (alpha/beta: exact-arithmetic fixtures)" begin
    @testset "mixed batch, T=$T" for T in (Float32, Float64, ComplexF64)
        # rank-2 transposes including one tensor above MINTHREADLENGTH elements (the
        # multi-threaded path when threads are enabled), then rank-3 transpose, payload and
        # identity permutations; more tensors than coefficients so the whole set cycles
        for (shapes, perm) in (
                ([(3, 4), (65, 33), (1, 7), (200, 200), (2, 2), (31, 1), (64, 64), (9, 8), (16, 16)], (2, 1)),
                ([(5, 6, 7), (33, 2, 31), (1, 1, 1), (2, 40, 3), (17, 17, 17), (1, 64, 2), (3, 3, 3), (8, 1, 8)], (3, 1, 2)),
                ([(5, 6, 7), (2, 33, 31), (1, 1, 32), (32, 2, 1)], (1, 3, 2)),
                ([(5, 6, 7), (33, 1, 2), (4, 4, 4)], (1, 2, 3)),
            )
            perm == (2, 1) && @test maximum(prod, shapes) > Strided.MINTHREADLENGTH
            srcs = [exactdata(T, s) for s in shapes]
            alpha, beta = scaledcoeffs(T, length(shapes))
            @test iszero(beta[1]) && iszero(alpha[3]) && any(!iszero, beta)
            scaledcase!(scaleddsts(T, srcs, perm, beta), srcs, perm, alpha, beta)
        end
    end

    @testset "omitted alpha/beta is exactly ones/zeros, T=$T" for T in (Float64, ComplexF64)
        shapes = [(7, 9), (65, 33), (1, 5), (200, 200)]
        perm = (2, 1)
        srcs = [exactdata(T, s) for s in shapes]
        B = length(srcs)
        alpha, beta = scaledcoeffs(T, B)
        beta = T[2, -1, 0.5, 1]          # all nonzero: the destination is read in every tensor
        d0 = [exactdata(T, dstshape(s, perm)) for s in srcs]
        runwith(a, b) = Strided.batched_permutedims!(map(copy, d0), srcs, perm; alpha = a, beta = b)
        # both omitted: the plain copy, and the same bits as explicit (ones, zeros)
        plain = runwith(nothing, nothing)
        for (d, s) in zip(plain, srcs)
            @test refequal(d, ref_base(s, perm))
        end
        @test all(map(refequal, plain, runwith(ones(T, B), zeros(T, B))))
        @test all(map(refequal, runwith(alpha, nothing), runwith(alpha, zeros(T, B))))
        @test all(map(refequal, runwith(nothing, beta), runwith(ones(T, B), beta)))
        # integer coefficient vectors convert to T
        @test all(map(refequal, runwith([2, -1, 0, 5], [1, 1, 0, 2]), runwith(T[2, -1, 0, 5], T[1, 1, 0, 2])))
        # a NaN-poisoned destination with beta omitted (== 0) comes out clean
        poisoned = [fill(T(NaN), dstshape(s, perm)) for s in srcs]
        Strided.batched_permutedims!(poisoned, srcs, perm; alpha)
        for (d, s, a) in zip(poisoned, srcs, alpha)
            @test refequal(d, a .* ref_base(s, perm))
        end
    end

    @testset "plan reuse with varying coefficients" begin
        T = Float64
        shapes = [(6, 7), (33, 65), (1, 1), (200, 200)]
        perm = (2, 1)
        srcs = [exactdata(T, s) for s in shapes]
        alpha, beta = scaledcoeffs(T, length(srcs))
        dsts = scaleddsts(T, srcs, perm, beta)
        plan = Strided.plan_batched_permutedims(dsts, srcs, perm)
        scaledcase!(dsts, srcs, perm, alpha, beta; plan)
        # the plan carries no coefficient state: other coefficients, and none at all, on reuse
        scaledcase!(dsts, srcs, perm, reverse(alpha), T[1, 0, -3, 0.5]; plan)
        scaledcase!(dsts, srcs, perm, T[0.5, 0.5, 0.5, 0.5], zeros(T, 4); plan)
        Strided.batched_permutedims!(dsts, srcs, plan)
        for (d, s) in zip(dsts, srcs)
            @test refequal(d, ref_base(s, perm))
        end
    end

    @testset "offset / strided / negative-stride views, destination read through the view" begin
        T = Float64
        perm = (2, 1)
        sph = [exactdata(T, (60, 60)) for _ in 1:4]
        dph = [exactdata(T, (60, 60)) for _ in 1:4]
        cases = (
            ((3:12, 5:20), (2:17, 7:16)),             # offset sub-blocks, dense
            ((3:2:21, 5:20), (2:3:47, 7:16)),         # strided
            ((12:-1:3, 5:20), (17:-1:2, 7:16)),       # reversed axis 1 on both sides
            ((21:-2:3, 20:-1:5), (2:17, 16:-1:7)),    # reversed non-fastest axes too
        )
        svs = [view(StridedView(p), sr...) for (p, (_, sr)) in zip(sph, cases)]
        dvs = [view(StridedView(p), dr...) for (p, (dr, _)) in zip(dph, cases)]
        alpha = T[2, -1, 0.5, 5]
        beta = T[0, 1, -3, 0]
        expected = map(copy, dph)
        for (e, (dr, sr), s, a, b) in zip(expected, cases, sph, alpha, beta)
            view(e, dr...) .= scaledref(a, view(s, sr...), b, view(e, dr...), perm)
        end
        out = Strided.batched_permutedims!(dvs, svs, perm; alpha, beta)
        @test out === dvs
        for (p, e) in zip(dph, expected)
            @test refequal(p, e)   # scaled bits inside the view, untouched outside it
        end
    end

    @testset "validation happens before any write" begin
        s = [exactdata(Float64, (3, 4)), exactdata(Float64, (5, 2))]
        poison() = [fill(NaN, 4, 3), fill(NaN, 2, 5)]
        plan = Strided.plan_batched_permutedims(poison(), s, (2, 1))
        for (alpha, beta) in ((ones(3), nothing), (nothing, zeros(1)), (ones(2), Float64[]), ([1.0], [0.0]))
            d = poison()
            @test_throws DimensionMismatch Strided.batched_permutedims!(d, s, (2, 1); alpha, beta)
            @test_throws DimensionMismatch Strided.batched_permutedims!(d, s, plan; alpha, beta)
            @test all(x -> all(isnan, x), d)
        end
        # a complex coefficient with nonzero imaginary part on a real batch is inexact
        d = poison()
        @test_throws InexactError Strided.batched_permutedims!(d, s, (2, 1); alpha = [1 + 2im, 1])
        @test_throws InexactError Strided.batched_permutedims!(d, s, plan; beta = [0, 0.5im])
        @test all(x -> all(isnan, x), d)
        # ... but a complex coefficient with zero imaginary part is fine
        scaledcase!(poison(), s, (2, 1), [2 + 0im, -1 + 0im], [0, 0])
        # coefficients need a Number element type; the plain copy does not
        st = [rand(NTuple{2, Float64}, 3, 4)]
        dt = [Array{NTuple{2, Float64}}(undef, 4, 3)]
        @test_throws ArgumentError Strided.batched_permutedims!(dt, st, (2, 1); alpha = [1.0])
        @test_throws ArgumentError Strided.batched_permutedims!(dt, st, (2, 1); beta = [0.0])
        runcase!(dt, st, (2, 1))
    end
end
