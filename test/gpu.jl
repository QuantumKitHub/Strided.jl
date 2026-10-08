test_result(a::AbstractArray, b::AbstractArray; kwargs...) =
    isapprox(Array(a), Array(b); kwargs...)
test_result(a::Number, b::Number; kwargs...) = isapprox(a, b; kwargs...)

function compare(f, AT::Type, xs...; kwargs...)
    cpu_in = map(deepcopy, xs) # copy on CPU
    gpu_in = map(adapt(AT), xs) # adapt on GPU

    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)
    return test_result(cpu_out, gpu_out; kwargs...)
end

# types to test for
ATs = []
!is_buildkite && push!(ATs, JLArray)
CUDACore.functional() && cuBLAS.functional() && push!(ATs, CuArray)
AMDGPU.functional() && push!(ATs, ROCArray)
Metal.functional() && push!(ATs, MtlArray)

@testset "isblasmatrix ($AT)" for AT in ATs
    for T in (Float32, ComplexF32)
        A1 = StridedView(AT(randn(T, 20, 20)))
        @test Strided.isblasmatrix(A1)
        A2 = view(A1, 1:4:20, 1:5:20)
        @test !Strided.isblasmatrix(A2)
        A3 = view(conj!(A1), 1:4:20, 1:20) # stride(A3, 2) is not 1
        @test !Strided.isblasmatrix(A3)
    end
end

@testset "in-place matrix operations ($AT)" for AT in ATs
    for T in (Float32, ComplexF32)
        A1 = StridedView(randn(T, 20, 20))
        A2 = StridedView(randn(T, 20, 20))

        @test compare(conj!, AT, A1)
        @test compare(adjoint!, AT, A1, A2)
        @test compare(transpose!, AT, A1, A2)
        @test compare((x, y) -> permutedims!(x, y, (2, 1)), AT, A1, A2)

        B1 = A1[4:4:end, 1:4:end]
        B2 = A2[4:4:end, 1:4:end]

        @test compare(conj!, AT, B1)
        @test compare(adjoint!, AT, B1, B2)
        @test compare(transpose!, AT, B1, B2)
        @test compare((x, y) -> permutedims!(x, y, (2, 1)), AT, B1, B2)
    end
end

@testset "mul! ($AT{$T})" for AT in ATs, T in (Float32, ComplexF32)
    N = 2
    α = rand(T)
    β = rand(T)
    dims = ntuple(Returns(div(64, N)), N)
    A1 = permutedims(StridedView(rand(T, dims)), randperm(N))
    A2 = permutedims(StridedView(rand(T, dims)), randperm(N))
    A3 = permutedims(StridedView(rand(T, dims)), randperm(N))
    @test compare((C, A, B) -> mul!(C, A, B, α, β), AT, A1, A2, A3)
    # test BLAS for all op combinations
    @testset for sz in ((32, 64), (64, 64), (64, 32))
        vA1 = view(StridedView(rand(T, sz)), 1:32, 1:32)
        vA2 = view(StridedView(rand(T, sz)), 1:32, 1:32)
        vA3 = view(StridedView(rand(T, sz)), 1:32, 1:32)
        @testset for f1 in (identity, conj, adjoint, transpose), f2 in (identity, conj, adjoint, transpose)
            @test compare((C, A, B) -> mul!(C, A, B, α, β), AT, vA1, f1(vA2), f2(vA3))
        end
    end
    # non-BLAS fallback path
    vA1 = view(StridedView(rand(T, (32, 32))), 1:32, 1:32)
    vA2 = view(StridedView(rand(T, (32, 64))), 1:32, 1:2:64)
    vA3 = view(StridedView(rand(T, (64, 32))), 1:2:64, 1:32)
    @testset for f1 in (identity, conj, adjoint, transpose), f2 in (identity, conj, adjoint, transpose)
        @test compare((C, A, B) -> mul!(C, A, B, α, β), AT, vA1, f1(vA2), f2(vA3))
    end
    # non-BLAS fallback path
    vA1 = view(StridedView(rand(T, (64, 32))), 1:2:64, 1:32)
    vA2 = view(StridedView(rand(T, (32, 64))), 1:32, 1:2:64)
    vA3 = view(StridedView(rand(T, (64, 32))), 1:2:64, 1:32)
    @test compare((C, A, B) -> mul!(C, A, B, α, β), AT, vA1, vA2, vA3)
end

@testset "map, scale!, axpy!, axpby! ($AT)" for AT in ATs
    for T in (Float32, ComplexF32)
        for N in 2:6
            dims = ntuple(Returns(div(60, N)), N)
            A1 = permutedims(StridedView(rand(T, dims)), randperm(N))
            A2 = permutedims(StridedView(rand(T, dims)), randperm(N))
            A3 = permutedims(StridedView(rand(T, dims)), randperm(N))

            @test compare(x -> rmul!(x, 1 // 2), AT, A1)
            @test compare(x -> lmul!(1 // 3, x), AT, A2)
            @test compare((x, y) -> axpy!(1 // 3, x, y), AT, A1, A2)
            @test compare((x, y) -> axpby!(1 // 3, x, 1 // 2, y), AT, A1, A2)
            @test compare((x, y, z) -> map((a, b, c) -> cos(a) + b / exp(-abs(c)), x, y, z), AT, A1, A2, A3)
            @test compare((x, y) -> mul!(x, 1, y), AT, A1, A2)
            @test compare((x, y) -> mul!(x, y, 1), AT, A1, A2)
        end

        dims = ntuple(Returns(20), 2)
        A1 = permutedims(StridedView(rand(T, dims))[2:2:end, 2:2:end], randperm(2))
        A2 = permutedims(StridedView(rand(T, dims))[2:2:end, 2:2:end], randperm(2))
        A3 = permutedims(StridedView(rand(T, dims))[2:2:end, 2:2:end], randperm(2))
        @test compare(x -> rmul!(x, 1 // 2), AT, A1)
        @test compare(x -> lmul!(1 // 3, x), AT, A2)
        @test compare((x, y) -> axpy!(1 // 3, x, y), AT, A1, A2)
        @test compare((x, y) -> axpby!(1 // 3, x, 1 // 2, y), AT, A1, A2)
        @test compare((x, y, z) -> map((a, b, c) -> cos(a) + b / exp(-abs(c)), x, y, z), AT, A1, A2, A3)
        @test compare((x, y) -> mul!(x, 1, y), AT, A1, A2)
        @test compare((x, y) -> mul!(x, y, 1), AT, A1, A2)
    end
end

@testset "copy ($AT)" for AT in ATs
    N = 2
    for m1 in (0, 16, 32), m2 in (0, 16, 32), T in (Float32, ComplexF32)
        dims = (m1, m2)
        A1 = StridedView(rand(T, dims))
        A2 = StridedView(rand(T, dims))
        A3 = StridedView(rand(T, dims))
        for f2 in (identity, conj, adjoint, transpose), f1 in (identity, conj, transpose, adjoint)
            axes(f1(A1)) == axes(f2(A2)) || continue
            B1 = f1(copy(A1))
            B2 = f2(copy(A2))
            @test compare((x, y) -> copy!(y, x), AT, B1, B2)
        end
    end
end

@testset "broadcasting ($AT)" for AT in ATs
    for T in (Float32, ComplexF32)
        A0 = StridedView(rand(T, ()))
        A1 = StridedView(rand(T, (10,)))
        A2 = permutedims(StridedView(rand(T, (10, 10))), randperm(2))
        A3 = permutedims(StridedView(rand(T, (10, 10, 10))), randperm(3))
        A4 = StridedView(rand(T, (2, 0)))

        @test compare((x, y) -> x .+ cos.(y .- 3), AT, A1, A2)
        @test compare((y, z) -> y' .* z .- Ref(1 // 2), AT, A2, A3)
        @test compare((x, y, z) -> y' .* z .- max.(abs.(x), real.(z)), AT, A1, A2, A3)
        @test compare((u, y, z) -> y' .* z .- u, AT, A0, A2, A3)

        @test compare(x -> x .+ x, AT, A4)
    end
end

@testset "mapreduce ($AT)" for AT in ATs
    sz = 10
    N = 6
    for T in (Float32, ComplexF32)
        A1 = StridedView(rand(T, ntuple(Returns(sz), N)))

        @test compare(x -> sum(x; dims = (1, 3, 5)), AT, A1)
        @test compare(x -> mapreduce(cos, +, x; dims = (1, 3, 5)), AT, A1)
        @test compare(x -> sum(x; dims = (1, 3, 5)), AT, permutedims(A1, randperm(N)))
        @test compare(x -> mapreduce(cos, +, x; dims = (1, 3, 5)), AT, permutedims(A1, randperm(N)))

        A2 = sreshape(StridedView(rand(T, ntuple(Returns(sz), 3))), (sz, 1, 1, sz, sz, 1))

        @test compare((x, y) -> Strided._mapreducedim!(cos, +, identity, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
        @test compare((x, y) -> Strided._mapreducedim!(cos, +, Returns(0), ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
        @test compare((x, y) -> Strided._mapreducedim!(cos, +, conj, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)

        β = rand(T)
        @test compare((x, y) -> Strided._mapreducedim!(cos, +, a -> β, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
        @test compare((x, y) -> Strided._mapreducedim!(cos, +, a -> β * a, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
    end
end

@testset "reduce ($AT)" for AT in ATs
    N = 4
    for T in (Float32, ComplexF32)
        A1 = StridedView(rand(T, ntuple(Returns(10), N)))
        A2 = permutedims(StridedView(rand(T, ntuple(Returns(10), N))), randperm(N))
        @test compare(sum, AT, A1)
        @test compare(sum, AT, A2)
        @test compare(x -> maximum(real, x), AT, A1)
        @test compare(x -> maximum(abs, x), AT, A2)
        @test compare(x -> minimum(abs, x), AT, A1)
        @test compare(x -> minimum(real, x), AT, A2)

        A3 = StridedView(rand(T, (5, 5, 5)))
        @test compare(x -> prod(exp, x), AT, A3)
    end
end

@testset "0-dimensional (scalar) StridedView ($AT)" for AT in ATs
    @testset for T in (Float32, ComplexF32)
        R = fill(rand(T)) # 0-dimensional Array
        A = StridedView(AT(R))
        @test ndims(A) == 0

        # full reductions
        @test sum(A) == sum(R)
        @test prod(A) == prod(R)
        @test mapreduce(abs2, +, A) == mapreduce(abs2, +, R)
        @test maximum(abs, A) == maximum(abs, R)
        @test minimum(abs, A) == minimum(abs, R)
        @test sum(abs2, A) == sum(abs2, R)
        @test mapreduce(identity, +, A; init = one(T)) ==
            mapreduce(identity, +, R; init = one(T))

        # map / map! / copy! / fill!
        mapx = map(x -> 2x, A)
        GPUArrays.@allowscalar begin
            @test mapx[] == 2 * R[]
        end
        B = StridedView(AT(fill(zero(T))))
        map!(x -> x + one(T), B, A)
        GPUArrays.@allowscalar begin
            @test B[] == collect(R)[] + one(T)
        end
        copy!(B, A)
        GPUArrays.@allowscalar begin
            @test B[] == R[]
        end
        fill!(B, one(T))
        GPUArrays.@allowscalar begin
            @test B[] == one(T)
        end

        # offset handling: 0-dim views into a larger parent
        Psrc = AT(rand(T, 5))
        Pdst = AT(rand(T, 5))
        s = sreshape(StridedView(Psrc)[4:4], ())
        d = sreshape(StridedView(Pdst)[3:3], ())
        GPUArrays.@allowscalar begin
            @test sum(s) == Psrc[4]
        end
        copy!(d, s)
        GPUArrays.@allowscalar begin
            @test Pdst[3] == Psrc[4]
        end

        # low-level in-place reduction with a custom initop
        Pd = AT(rand(T, 5))
        d2 = sreshape(StridedView(Pd)[2:2], ())
        GPUArrays.@allowscalar begin
            prev = Pd[2]
        end
        Strided._mapreducedim!(cos, +, identity, (), (d2, A))
        GPUArrays.@allowscalar begin
            @test Pd[2] == prev + cos(R[])
        end
    end
end

# ---------- batched out-of-place permutation: GPU execution strategies ----------
#
# Correctness of every `BatchedPermuteStrategy` on every available GPU array type, each
# checked bit-exactly (`isequal`, never `isapprox`: this operation moves bits and performs
# no arithmetic) against `permutedims` on the CPU copy of the same data, plus the strategy
# resolution table (`BP_AUTO` -> elementwise baseline; `BP_GROUPTILE` only for the
# transpose family, otherwise a silent, inspectable fallback to `BP_ELEMENTWISE`).
#
# Note on families: the planner never merges axes, so the copy family (`BP_COPY`, reduced
# rank 1) only arises for rank-1 batches; an identity permutation on a rank >= 2 batch is
# the payload family (`BP_PAYLOAD`). Both are exercised below.

const BP_STRATEGIES = (Strided.BP_AUTO, Strided.BP_ELEMENTWISE, Strided.BP_THREADTILE, Strided.BP_GROUPTILE)

bp_refequal(a::AbstractArray, b::AbstractArray) = isequal(Array(a), Array(b))

# what the resolution table says a request must resolve to, given the family
# Mirrors the resolver's own contract independently (family AND shared-memory-budget
# clauses both restated), so this stays a check on the resolution table rather than a
# restatement of only part of it -- a lower `_BP_GROUPTILE_SHMEM_BUDGET` must correctly
# flip this function's answer too, or a budget-related regression would go unnoticed. The
# budget clause is "some menu geometry's tile fits", not "the largest one does": an element
# type whose tile only fits at a smaller edge still runs the cooperative kernel, on that edge.
function bp_expected_strategy(requested, family, ::Type{T}) where {T}
    requested === Strided.BP_AUTO && return Strided.BP_ELEMENTWISE
    if requested === Strided.BP_GROUPTILE
        ext = Base.get_extension(Strided, :StridedGPUArraysExt)
        pad, budget = ext._BP_GROUPTILE_PAD, ext._BP_GROUPTILE_SHMEM_BUDGET
        fits = any(g -> sizeof(T) * (g.edge + pad) * g.edge <= budget, ext._BP_GROUPTILE_GEOMETRIES)
        (family !== Strided.BP_TRANSPOSE || !fits) && return Strided.BP_ELEMENTWISE
    end
    return requested
end

# The tile a `BP_GROUPTILE` plan must carry, restated independently of the extension's own
# selection code: over the plan's per-tensor reduced extents, each menu edge `e` schedules
# `cld(d1, e) * cld(dc, e) * e^2` thread slots per tensor (times the product of the other
# extents) for `d1 * dc * ...` real elements; the edge with the highest ratio wins among the
# edges whose tile fits the element type's local-memory budget, a tie going to the larger
# edge. That edge sits on the source-fastest axis 1 and the destination-fastest axis
# `c = plan.dstseq[1]`, 1 elsewhere.
function bp_expected_grouptile_tile(plan::Strided.BatchedPermutePlan{N0, N, T}) where {N0, N, T}
    ext = Base.get_extension(Strided, :StridedGPUArraysExt)
    pad, budget = ext._BP_GROUPTILE_PAD, ext._BP_GROUPTILE_SHMEM_BUDGET
    c = plan.dstseq[1]
    dims = [d.dims for d in plan.descs]
    elements = sum(prod, dims; init = 0)
    slots(e) = sum(prod(g == 1 || g == c ? cld(d[g], e) : d[g] for g in 1:N) * e * e for d in dims; init = 0)
    best, bestu = 0, -1.0
    for g in ext._BP_GROUPTILE_GEOMETRIES
        sizeof(T) * (g.edge + pad) * g.edge <= budget || continue
        s = slots(g.edge)
        u = s == 0 ? 1.0 : elements / s
        u > bestu && ((best, bestu) = (g.edge, u))
    end
    return ntuple(g -> (g == 1 || g == c) ? best : 1, N)
end

function bp_fixture(AT, T, shapes, perm)
    srcs_cpu = [rand(T, s...) for s in shapes]
    srcs = [AT(s) for s in srcs_cpu]
    dsts = [AT(zeros(T, ntuple(j -> size(s, perm[j]), length(perm)))) for s in srcs_cpu]
    refs = [permutedims(s, perm) for s in srcs_cpu]
    return srcs_cpu, srcs, dsts, refs
end

# plan with `strategy`, run, check the return value, bit-exactness, unmodified sources, the
# reported family, and that the resolved strategy is exactly what the table prescribes.
function bp_runcase(AT, T, shapes, perm, strategy, family)
    srcs_cpu, srcs, dsts, refs = bp_fixture(AT, T, shapes, perm)
    plan = Strided.plan_batched_permutedims(dsts, srcs, perm; strategy)
    @test plan.family === family
    @test plan.strategy === bp_expected_strategy(strategy, family, T)
    out = Strided.batched_permutedims!(dsts, srcs, plan)
    @test out === dsts
    for (d, r) in zip(dsts, refs)
        @test bp_refequal(d, r)
    end
    for (s, s0) in zip(srcs, srcs_cpu)
        @test bp_refequal(s, s0)
    end
    return plan
end

@testset "batched_permutedims! GPU strategies ($AT)" for AT in ATs
    Ext = Base.get_extension(Strided, :StridedGPUArraysExt)
    EDGE = Ext._BP_GROUPTILE_GEOMETRIES[1].edge

    @testset "strategy=$strategy" for strategy in BP_STRATEGIES
        # transpose family, extents deliberately not multiples of 32 (edge tiles), both a
        # 4-byte and a 16-byte element type
        for T in (Float32, ComplexF64)
            plan = bp_runcase(AT, T, [(67, 51), (33, 32), (1, 64), (31, 2), (32, 32)], (2, 1),
                strategy, Strided.BP_TRANSPOSE)
            if plan.strategy === Strided.BP_ELEMENTWISE
                @test all(==(1), plan.tile)          # BP_AUTO must be the unchanged baseline
            elseif plan.strategy === Strided.BP_THREADTILE
                @test count(!=(1), plan.tile) == 1   # exactly one tiled axis
                @test maximum(plan.tile) == Ext._BP_THREADTILE_K[]
            elseif plan.strategy === Strided.BP_GROUPTILE
                # which menu edge: the utilization rule on this batch's own extents (the
                # edges themselves are pinned by hand in the geometry-menu testset)
                @test plan.tile == bp_expected_grouptile_tile(plan)
                @test count(!=(1), plan.tile) == 2
            end
        end
        # payload family (rank 3, shared contiguous axis 1): GROUPTILE must fall back
        bp_runcase(AT, Float32, [(5, 6, 7), (2, 33, 31), (1, 1, 32), (32, 2, 1)], (1, 3, 2),
            strategy, Strided.BP_PAYLOAD)
        # identity permutation on a rank-3 batch: also payload family
        bp_runcase(AT, Float32, [(5, 6, 7), (33, 1, 2)], (1, 2, 3), strategy, Strided.BP_PAYLOAD)
        # copy family (rank 1): GROUPTILE must fall back
        bp_runcase(AT, Float32, [(5,), (33,), (1,), (64,), (31,)], (1,), strategy, Strided.BP_COPY)
        # rank 3 and rank 4 transposes with edge-adjacent extents 1, 2, 31, 32, 33 mixed in
        plan3 = bp_runcase(AT, Float32, [(33, 2, 31), (1, 32, 33), (2, 1, 1), (31, 33, 2)], (3, 1, 2),
            strategy, Strided.BP_TRANSPOSE)
        plan4 = bp_runcase(AT, Float32, [(32, 1, 33, 2), (31, 2, 1, 32), (1, 33, 2, 31), (2, 2, 2, 2)],
            (4, 2, 1, 3), strategy, Strided.BP_TRANSPOSE)
        if strategy === Strided.BP_GROUPTILE
            for p in (plan3, plan4)
                @test p.tile == bp_expected_grouptile_tile(p)
                @test p.tile[1] in map(g -> g.edge, Ext._BP_GROUPTILE_GEOMETRIES)
                @test p.tile[p.dstseq[1]] == p.tile[1] && count(==(p.tile[1]), p.tile) == 2
            end
        end
        # one dominant tensor plus many tiny ones (tile ids straddle many prefix boundaries)
        shapes = Tuple{Int, Int}[(129, 97)]
        append!(shapes, [(3, 2) for _ in 1:20])
        append!(shapes, [(1, 1) for _ in 1:5])
        bp_runcase(AT, Float32, shapes, (2, 1), strategy, Strided.BP_TRANSPOSE)
    end

    @testset "plan reuse keeps its resolved strategy" begin
        for strategy in (Strided.BP_THREADTILE, Strided.BP_GROUPTILE)
            shapes = [(67, 51), (33, 32), (1, 5)]
            _, srcs1, dsts1, _ = bp_fixture(AT, Float32, shapes, (2, 1))
            plan = Strided.plan_batched_permutedims(dsts1, srcs1, (2, 1); strategy)
            @test plan.strategy === strategy
            for _ in 1:2
                srcs_cpu, srcs, dsts, refs = bp_fixture(AT, Float32, shapes, (2, 1))
                Strided.batched_permutedims!(dsts, srcs, plan)
                for (d, r) in zip(dsts, refs)
                    @test bp_refequal(d, r)
                end
                for (s, s0) in zip(srcs, srcs_cpu)
                    @test bp_refequal(s, s0)
                end
            end
            @test plan.strategy === strategy
        end
    end

    @testset "plan reuse re-checks strides, not just addresses/offsets" begin
        # Regression: a device-binding cache keyed only on base address + offset would
        # wrongly reuse stale, previously-uploaded strides for a second call that shares
        # the same parent, same offset, but a genuinely different view (here: every-other
        # column) into it -- producing a silently wrong result with no error raised.
        for strategy in (Strided.BP_ELEMENTWISE, Strided.BP_GROUPTILE)
            Aparent = AT(Float32.(reshape(1:64, 8, 8)))
            s1 = view(StridedView(Aparent), 1:4, 1:4)     # dense 4x4 block, strides (1,8)
            s2 = view(StridedView(Aparent), 1:4, 1:2:8)   # same base & offset, strides (1,16)
            ref1 = permutedims(Array(Aparent)[1:4, 1:4], (2, 1))
            ref2 = permutedims(Array(Aparent)[1:4, 1:2:8], (2, 1))
            dst = AT(zeros(Float32, 4, 4))
            plan = Strided.plan_batched_permutedims([dst], [s1], (2, 1); strategy)
            Strided.batched_permutedims!([dst], [s1], plan)
            @test bp_refequal(dst, ref1)
            fill!(dst, 0)
            Strided.batched_permutedims!([dst], [s2], plan)
            @test bp_refequal(dst, ref2)
        end
    end

    @testset "BP_THREADTILE orientation knob" begin
        along0 = Ext._BP_THREADTILE_ALONG[]
        try
            Ext._BP_THREADTILE_ALONG[] = :srcfast
            plan = bp_runcase(AT, Float32, [(67, 51), (3, 70)], (2, 1), Strided.BP_THREADTILE, Strided.BP_TRANSPOSE)
            @test plan.tile == (Ext._BP_THREADTILE_K[], 1)
            # axis 1's largest extent here (5) is below K, so the tile is clamped to it
            plan = bp_runcase(AT, Float32, [(5, 6, 7), (2, 33, 31)], (1, 3, 2), Strided.BP_THREADTILE, Strided.BP_PAYLOAD)
            @test plan.tile == (min(Ext._BP_THREADTILE_K[], 5), 1, 1)
        finally
            Ext._BP_THREADTILE_ALONG[] = along0
        end
        plan = bp_runcase(AT, Float32, [(5, 6, 7), (2, 33, 31)], (1, 3, 2), Strided.BP_THREADTILE, Strided.BP_PAYLOAD)
        @test plan.tile == (1, Ext._BP_THREADTILE_K[], 1)
    end

    @testset "resolution table" begin
        backend = GPUArrays.KernelAbstractions.get_backend(AT(zeros(Float32, 1)))
        resolve(req, fam, T) = Strided._batched_resolve_strategy(backend, req, fam, T)
        for fam in (Strided.BP_COPY, Strided.BP_PAYLOAD, Strided.BP_TRANSPOSE), T in (Float32, ComplexF64)
            @test resolve(Strided.BP_AUTO, fam, T) === Strided.BP_ELEMENTWISE
            @test resolve(Strided.BP_ELEMENTWISE, fam, T) === Strided.BP_ELEMENTWISE
            @test resolve(Strided.BP_THREADTILE, fam, T) === Strided.BP_THREADTILE
            @test resolve(Strided.BP_GROUPTILE, fam, T) ===
                (fam === Strided.BP_TRANSPOSE ? Strided.BP_GROUPTILE : Strided.BP_ELEMENTWISE)
        end
        # an element type whose 32x33 tile exceeds the local-memory budget stays eligible as
        # long as the menu's smallest tile fits it (it then runs on that geometry); only an
        # element type for which no menu tile fits falls back
        small = minimum(g -> g.edge, Ext._BP_GROUPTILE_GEOMETRIES)
        Tbig = NTuple{8, Float64}
        Tnone = NTuple{16, Float64}
        @test sizeof(Tbig) * (EDGE + Ext._BP_GROUPTILE_PAD) * EDGE > Ext._BP_GROUPTILE_SHMEM_BUDGET
        @test sizeof(Tbig) * (small + Ext._BP_GROUPTILE_PAD) * small <= Ext._BP_GROUPTILE_SHMEM_BUDGET
        @test sizeof(Tnone) * (small + Ext._BP_GROUPTILE_PAD) * small > Ext._BP_GROUPTILE_SHMEM_BUDGET
        @test resolve(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, Tbig) === Strided.BP_GROUPTILE
        @test resolve(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, Tnone) === Strided.BP_ELEMENTWISE
        @test resolve(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, ComplexF64) === Strided.BP_GROUPTILE
    end

    @testset "BP_GROUPTILE tile-shape invariant is enforced at launch" begin
        _, srcs, dsts, _ = bp_fixture(AT, Float32, [(67, 51)], (2, 1))
        plan = Strided.plan_batched_permutedims(dsts, srcs, (2, 1); strategy = Strided.BP_GROUPTILE)
        @test plan.strategy === Strided.BP_GROUPTILE
        # same plan, but with a tile shape the cooperative kernel does not implement
        P = typeof(plan)
        bad = P(plan.perm, plan.srcorder, plan.groups, plan.dstseq, plan.family, plan.strategy,
            ntuple(Returns(1), length(plan.tile)), plan.srclabels, plan.dstlabels, plan.descs,
            plan.cpublocks, plan.prefix, plan.totaltiles, plan.totalelements, plan.deviceid,
            Base.RefValue{Any}(nothing))
        @test_throws ArgumentError Strided.batched_permutedims!(dsts, srcs, bad)
    end
end

# ---------- batched out-of-place permutation: the device-binding cache ----------
#
# The GPU executor caches its per-plan device upload (descriptors, prefix sums, base
# addresses) in `plan.devcache` and reuses it only if the device identity, EVERY parent
# array's current device address, and every host-side descriptor still match. These tests
# pin both halves of that contract: a genuine repeat call reuses the very same cached
# binding (and is bit-exact), and anything that moves data to a different address -- a
# swapped source, a swapped destination, or a `resize!` that relocates a buffer behind an
# unchanged array object -- invalidates it, so the result reflects the data now at the
# arrays actually passed in, never a stale upload. The swapped-array cases are built so
# that the planner's own reuse check cannot be what catches them: a fresh array of the same
# shape has identical strides and offset, hence a bitwise-identical descriptor, so the plan
# object itself is reused as-is and only the address re-check stands between the call and
# a silently wrong answer.
@testset "device binding cache: reuse and invalidation ($AT)" for AT in ATs
    Ext = Base.get_extension(Strided, :StridedGPUArraysExt)
    shapes = [(33, 17), (5, 40), (16, 16), (2, 65)]
    perm = (2, 1)
    for strategy in (Strided.BP_ELEMENTWISE, Strided.BP_GROUPTILE)
        @testset "strategy=$strategy" begin
            srcs_cpu, srcs, dsts, refs = bp_fixture(AT, Float32, shapes, perm)
            plan = Strided.plan_batched_permutedims(dsts, srcs, perm; strategy)
            @test plan.devcache[] === nothing
            Strided.batched_permutedims!(dsts, srcs, plan)
            b1 = plan.devcache[]
            @test b1 isa Ext._BatchedGPUBinding
            for (d, r) in zip(dsts, refs)
                @test bp_refequal(d, r)
            end

            # (a) the same arrays again: the same cached binding object, bit-exact results
            for d in dsts
                fill!(d, NaN32)
            end
            Strided.batched_permutedims!(dsts, srcs, plan)
            @test plan.devcache[] === b1
            for (d, r) in zip(dsts, refs)
                @test bp_refequal(d, r)
            end
            # the planner's reuse check indeed hands back the same plan object here, so the
            # binding cache's own address check is the only thing guarding the cases below
            dviews, sviews, _ = Strided._normalize_and_check(dsts, srcs, 2)
            @test Strided._rebuild_descriptors(dviews, sviews, plan) === plan

            # (b) one source replaced by a fresh same-shape array, the old one overwritten
            # with a sentinel: the result must show the new array's data
            oldsrc = srcs[2]
            newsrc_cpu = rand(Float32, shapes[2]...)
            srcs[2] = AT(newsrc_cpu)
            fill!(oldsrc, -1.0f0)
            for d in dsts
                fill!(d, NaN32)
            end
            dviews, sviews, _ = Strided._normalize_and_check(dsts, srcs, 2)
            @test Strided._rebuild_descriptors(dviews, sviews, plan) === plan # same descriptors
            Strided.batched_permutedims!(dsts, srcs, plan)
            @test plan.devcache[] !== b1
            @test bp_refequal(dsts[2], permutedims(newsrc_cpu, perm))
            @test !any(==(-1.0f0), Array(dsts[2]))
            for i in (1, 3, 4)
                @test bp_refequal(dsts[i], refs[i])
            end
            @test all(==(-1.0f0), Array(oldsrc)) # sources are never written

            # (b') the same for a destination: the new one gets written, the old one (now
            # holding a sentinel) is left alone
            b2 = plan.devcache[]
            olddst = dsts[3]
            dsts[3] = AT(fill(NaN32, size(olddst)))
            fill!(olddst, -2.0f0)
            Strided.batched_permutedims!(dsts, srcs, plan)
            @test plan.devcache[] !== b2
            @test bp_refequal(dsts[3], refs[3])
            @test all(==(-2.0f0), Array(olddst))

            # (c) a repeat call after the swaps is a cache hit again
            b3 = plan.devcache[]
            Strided.batched_permutedims!(dsts, srcs, plan)
            @test plan.devcache[] === b3
        end
    end

    if AT === CuArray
        @testset "resize! relocating a CuVector behind the same array object" begin
            # `resize!` on a CuVector allocates a new buffer and frees the old one, even when
            # growing and then shrinking back to the original length -- the array object,
            # its length and its shape are all unchanged afterwards, but its data lives at
            # a new address (observed to relocate on every trial in this environment; the
            # address is still checked below rather than assumed). An identity- or
            # length-based cache shortcut would keep using the freed old buffer here.
            a_cpu = rand(Float32, 1000)
            a = AT(a_cpu)
            d = AT(zeros(Float32, 1000))
            plan = Strided.plan_batched_permutedims([d], [a], (1,))
            Strided.batched_permutedims!([d], [a], plan)
            @test bp_refequal(d, a_cpu)
            b1 = plan.devcache[]
            addr0 = UInt(pointer(a))
            resize!(a, 2000)
            resize!(a, 1000)
            addr1 = UInt(pointer(a))
            @test length(a) == 1000
            new_cpu = rand(Float32, 1000)
            copyto!(a, new_cpu)
            fill!(d, NaN32)
            Strided.batched_permutedims!([d], [a], plan)
            @test bp_refequal(d, new_cpu)
            # only meaningful as a cache-invalidation check if the buffer actually moved;
            # the correctness assertion above holds either way
            if addr1 != addr0
                @test plan.devcache[] !== b1
            end
        end
    end
end

# ---------- batched out-of-place permutation: addressing under views and deep lookups ----------
#
# The cooperative `BP_GROUPTILE` kernel derives every address from per-tensor descriptors
# (offset, strides along the two active axes, tile extents) that travel from its load phase
# to its store phase through workgroup-local memory, and finds its tensor by a binary search
# over the batch's tile-id prefix sums. The checks below exercise exactly those paths, for
# `BP_GROUPTILE` and (as the reference implementation of the same addressing) the
# elementwise baseline: a batch deep enough that the search has real depth, extents that sit
# on, just inside and just outside a tile edge along both active axes, and source *and*
# destination views with offsets, non-unit strides and negative strides -- each with a guard
# band, i.e. the whole destination parent compared against a sentinel-filled reference so
# that a write landing anywhere outside the view is caught, not just a wrong value inside it.

const BP_EDGE_EXTENTS = (1, 2, 31, 32, 33, 63, 64, 65)

# One batch of views: `cases` is a vector of `(dranges, sranges)` pairs, each carving a
# destination view and a source view out of its own freshly allocated parent pair. Returns
# the plan so the caller can assert what strategy the request actually resolved to.
function bp_viewcase(AT, T, perm, dparentsize, sparentsize, cases, strategy)
    dphs = [fill(T(NaN), dparentsize...) for _ in cases]
    sphs = [rand(T, sparentsize...) for _ in cases]
    dps = [AT(p) for p in dphs]
    sps = [AT(p) for p in sphs]
    dvs = [view(StridedView(dp), dr...) for (dp, (dr, _)) in zip(dps, cases)]
    svs = [view(StridedView(sp), sr...) for (sp, (_, sr)) in zip(sps, cases)]
    for (dv, sv) in zip(dvs, svs)
        @test size(dv) == ntuple(j -> size(sv, perm[j]), length(perm))
    end
    plan = Strided.plan_batched_permutedims(dvs, svs, perm; strategy)
    out = Strided.batched_permutedims!(dvs, svs, plan)
    @test out === dvs
    for (dp, sp, dph, sph, (dr, sr)) in zip(dps, sps, dphs, sphs, cases)
        expected = copy(dph)
        view(expected, dr...) .= permutedims(view(sph, sr...), perm)
        @test isequal(Array(dp), expected)   # right bits inside the view, sentinel outside
        @test isequal(Array(sp), sph)        # source parent untouched
    end
    return plan
end

@testset "batched_permutedims! GPU addressing: views, edge extents, deep lookup ($AT)" for AT in ATs
    Ext = Base.get_extension(Strided, :StridedGPUArraysExt)
    EDGE = Ext._BP_GROUPTILE_GEOMETRIES[1].edge

    @testset "deep tile->tensor lookup, strategy=$strategy" for strategy in BP_STRATEGIES
        # >= 300 tensors of mixed extents: the prefix array has > 2^8 entries, so the
        # device-side binary search runs to depth 9, and consecutive tile ids constantly
        # cross from one tensor to the next (most tensors here are a single, partial tile).
        cycle = [(5, 7), (33, 31), (1, 64), (64, 1), (2, 2), (31, 33), (65, 3), (3, 65),
            (32, 32), (17, 9), (1, 1), (2, 33), (33, 2), (64, 64)]
        shapes = [cycle[mod1(i, length(cycle))] for i in 1:320]
        plan = bp_runcase(AT, Float32, shapes, (2, 1), strategy, Strided.BP_TRANSPOSE)
        @test length(plan.prefix) == 321
        @test plan.totaltiles == sum(prod(cld.(s, plan.tile)) for s in shapes)
    end

    @testset "edge extents on both active axes, strategy=$strategy" for strategy in
            (Strided.BP_ELEMENTWISE, Strided.BP_GROUPTILE)
        # every pair of extents from {1, 2, 31, 32, 33, 63, 64, 65} on the source-fastest and
        # destination-fastest axes, i.e. tiles that are exactly full, one element short, one
        # element over, and multi-tile in each direction, for rank 2 and for rank 3 with the
        # destination-fastest axis being source axis 3 (perm (3,1,2)) or 2 (perm (2,3,1))
        grid = [(i, j) for i in BP_EDGE_EXTENTS for j in BP_EDGE_EXTENTS]
        plan2 = bp_runcase(AT, Float32, grid, (2, 1), strategy, Strided.BP_TRANSPOSE)
        m = Iterators.cycle((1, 2, 3))
        grid3a = [(i, mm, j) for ((i, j), mm) in zip(grid, m)]
        plan3a = bp_runcase(AT, Float32, grid3a, (3, 1, 2), strategy, Strided.BP_TRANSPOSE)
        grid3b = [(i, j, mm) for ((i, j), mm) in zip(grid, m)]
        plan3b = bp_runcase(AT, Float32, grid3b, (2, 3, 1), strategy, Strided.BP_TRANSPOSE)
        if strategy === Strided.BP_GROUPTILE
            @test plan2.tile == bp_expected_grouptile_tile(plan2)
            @test plan3a.tile == bp_expected_grouptile_tile(plan3a) && plan3a.dstseq[1] == 3
            @test plan3b.tile == bp_expected_grouptile_tile(plan3b) && plan3b.dstseq[1] == 2
        end
    end

    @testset "offset / strided / negative-stride views with guard band, strategy=$strategy" for
            strategy in (Strided.BP_ELEMENTWISE, Strided.BP_GROUPTILE)
        T = Float32
        P = (210, 210)
        # Rank 2, perm (2,1): destination views are 40 x 67, source views 67 x 40. In every
        # case here the view's fastest axis is still its axis 1 (positive or negative unit
        # or non-unit stride), so the batch is the transpose family and a `BP_GROUPTILE`
        # request must actually run the cooperative kernel, not fall back.
        for (dr, sr) in (
                ((3:42, 5:71), (2:68, 7:46)),             # offset sub-blocks, dense
                ((3:2:81, 5:71), (2:68, 7:46)),           # stride-2 destination axis 1
                ((3:42, 5:71), (2:3:200, 7:46)),          # stride-3 source axis 1
                ((42:-1:3, 5:71), (2:68, 7:46)),          # reversed destination axis 1
                ((3:42, 5:71), (68:-1:2, 7:46)),          # reversed source axis 1
                ((81:-2:3, 5:71), (200:-3:2, 7:46)),      # both reversed and strided
                ((3:42, 7:2:139), (2:68, 7:46)),          # stride-2 destination axis 2
            )
            plan = bp_viewcase(AT, T, (2, 1), P, P, [(dr, sr)], strategy)
            @test plan.family === Strided.BP_TRANSPOSE
            @test plan.strategy === strategy
        end
        # Several views in one batch, each from its own parent pair, mixing the variants
        # above so that per-tensor offsets and strides genuinely differ across the batch.
        plan = bp_viewcase(AT, T, (2, 1), P, P, [
                ((3:42, 5:71), (2:68, 7:46)),
                ((81:-2:3, 5:71), (2:3:200, 7:46)),
                ((42:-1:3, 5:71), (68:-1:2, 7:46)),
                ((100:139, 7:73), (50:116, 150:189)),
            ], strategy)
        @test plan.family === Strided.BP_TRANSPOSE
        @test plan.strategy === strategy
        # A 16-byte element type through the same path (one case, to bound compile time).
        plan = bp_viewcase(AT, ComplexF64, (2, 1), P, P, [((81:-2:3, 5:71), (200:-3:2, 7:46))], strategy)
        @test plan.strategy === strategy
        # Rank 3, perm (3,1,2): destination 34 x 31 x 3, source 31 x 3 x 34, with an offset
        # source and a reversed destination axis 1.
        P3 = (40, 40, 40)
        plan = bp_viewcase(AT, T, (3, 1, 2), P3, P3,
            [((37:-1:4, 5:35, 2:4), (2:32, 5:7, 3:36)), ((4:37, 5:35, 2:4), (32:-1:2, 5:7, 3:36))],
            strategy)
        @test plan.family === Strided.BP_TRANSPOSE
        @test plan.strategy === strategy
        # A negative stride on a view's *non*-fastest axis flips the planner's heuristic
        # axis order (it sorts by signed stride), which reclassifies the batch as the
        # payload family and makes a `BP_GROUPTILE` request fall back to the elementwise
        # kernel. The results must still be bit-exact either way; which strategy actually
        # ran is deliberately not asserted here, so a later change to that heuristic does
        # not turn this into a false failure.
        for (dr, sr) in (
                ((3:42, 5:71), (2:68, 46:-1:7)),          # reversed source axis 2
                ((3:42, 71:-1:5), (2:68, 7:46)),          # reversed destination axis 2
            )
            plan = bp_viewcase(AT, T, (2, 1), P, P, [(dr, sr)], strategy)
            @test plan.strategy in (Strided.BP_ELEMENTWISE, Strided.BP_GROUPTILE)
        end
    end
end

# ---------- batched out-of-place permutation: BP_GROUPTILE tile-geometry menu ----------
#
# `BP_GROUPTILE` picks its tile edge from a small fixed menu (`_BP_GROUPTILE_GEOMETRIES`),
# so that a batch of tensors much smaller than a 32x32 tile can run on a 16x16 one instead of
# idling three quarters of every workgroup. The edge is chosen per batch by the exact
# tile-slot utilization rule over every tensor's own extents (so a batch mixing a few large
# tensors with many tiny ones is weighed tensor by tensor), restricted to the edges whose
# tile fits the element type's local-memory budget. The checks below pin down: the menu's
# shape (bounded, ordered, divisibility), the utilization rule on hand-computable synthetic
# extents, the resolver's "some edge fits" eligibility, what the tile-shape hook actually
# selects (asserted on `plan.tile` against hand-computed slot counts AND against
# `bp_expected_grouptile_tile`) for small-only, mixed, tie and large batches, bit-exact runs
# through both geometries on full and partial tiles (Float32, rank 2 and 3 only, to bound
# the number of compiled kernel variants; one 64-byte element type on the small geometry),
# and the executor's rejection of any tile edge that is not a menu entry or does not fit the
# element type.

@testset "batched_permutedims! GPU BP_GROUPTILE geometry menu ($AT)" for AT in ATs
    Ext = Base.get_extension(Strided, :StridedGPUArraysExt)
    menu = Ext._BP_GROUPTILE_GEOMETRIES
    EDGE = Ext._BP_GROUPTILE_GEOMETRIES[1].edge
    edges = map(g -> g.edge, menu)
    budget = Ext._BP_GROUPTILE_SHMEM_BUDGET
    tilebytes = Ext._batched_grouptile_tilebytes
    # a 64-byte element type: its 32x33 tile is over the budget, its 16x17 tile is not;
    # and a 128-byte one for which no menu tile fits at all
    Tbig = NTuple{8, Float64}
    Tnone = NTuple{16, Float64}
    @test tilebytes(Tbig, 16) <= budget < tilebytes(Tbig, EDGE)
    @test tilebytes(Tnone, 16) > budget
    bigpoison = ntuple(Returns(NaN), 8)   # `zeros`/`T(NaN)` do not exist for tuple types

    @testset "menu invariants" begin
        @test 1 <= length(menu) <= 3
        @test menu[1] == (edge = 32, rows = 4)
        for (i, geom) in enumerate(menu)
            @test geom.edge >= 1 && geom.rows >= 1
            @test geom.edge % geom.rows == 0          # full (li, lj) coverage by EDGE÷ROWS rows per thread
            i > 1 && @test geom.edge < menu[i - 1].edge  # strictly decreasing: entry 1 is the largest
            # a smaller edge never needs more local memory than the largest one
            for T in (Float32, Float64, ComplexF64)
                @test Ext._batched_grouptile_tilebytes(T, geom.edge) <=
                    Ext._batched_grouptile_tilebytes(T, EDGE)
            end
        end
        # the checks below assume the shipped menu: 32 first, 16 present
        @test EDGE == 32
        @test 16 in edges
    end

    @testset "utilization rule on synthetic extents" begin
        util = Ext._batched_grouptile_utilization
        pick(dims, c, T = Float32) = menu[Ext._batched_grouptile_geometry(dims, c, T)].edge
        # one 16x16 tensor: a 32x32 tile is a quarter full, a 16x16 tile is exactly full
        @test util([(16, 16)], 2, 32) == 0.25
        @test util([(16, 16)], 2, 16) == 1.0
        # rank 3, active axes 1 and 3: the middle axis multiplies the tile count, not the waste
        @test util([(16, 16, 16)], 3, 32) == 0.25
        @test util([(16, 16, 16)], 3, 16) == 1.0
        # 33x33: 4 tiles of 32x32 (1089/4096) vs 9 tiles of 16x16 (1089/2304)
        @test util([(33, 33)], 2, 32) == 1089 / 4096
        @test util([(33, 33)], 2, 16) == 1089 / 2304
        # no tiles at all (empty batch, or every tensor empty): defined, and a tie
        @test util(NTuple{2, Int}[], 2, 32) == 1.0
        @test util([(0, 16), (16, 0)], 2, 16) == 1.0
        # which geometry wins, by hand
        @test pick([(16, 16) for _ in 1:2000], 2) == 16
        @test pick([(16, 16, 16) for _ in 1:2000], 3) == 16
        @test pick([(8, 8), (16, 1), (1, 16), (2, 3)], 2) == 16
        @test pick([(33, 33)], 2) == 16                    # finer edge idles fewer slots
        @test pick([(64, 64) for _ in 1:10], 2) == 32      # full tiles either way: tie -> largest
        @test pick([(4096, 4096)], 2) == 32
        @test pick([(20, 20)], 2) == 32                    # 1x1 tiles of 32 vs 2x2 of 16: same slots, tie
        @test pick(NTuple{2, Int}[], 2) == 32
        @test pick([(0, 16)], 2) == 32
        # mixed: one full-tile tensor plus many small ones -- the small ones dominate the slots
        @test pick(vcat([(64, 64)], [(16, 16) for _ in 1:1000]), 2) == 16
        # the other active axis is `c`, not always 2
        @test pick([(16, 100, 16)], 3) == 16
        @test pick([(16, 16, 100)], 2) == 16
        @test pick([(32, 100, 32)], 3) == 32
        # an element type whose tile only fits at the small edge gets that edge whatever the
        # extents (even on full 32x32 tiles, and on the no-tile tie), and one for which no
        # menu tile fits is refused outright rather than silently given a geometry
        @test pick([(4096, 4096)], 2, Tbig) == 16
        @test pick([(64, 64) for _ in 1:10], 2, Tbig) == 16
        @test pick(NTuple{2, Int}[], 2, Tbig) == 16
        @test_throws ArgumentError pick([(16, 16)], 2, Tnone)
    end

    @testset "resolver: eligible as soon as some menu geometry fits the local-memory budget" begin
        backend = GPUArrays.KernelAbstractions.get_backend(AT(zeros(Float32, 1)))
        resolve(req, fam, T) = Strided._batched_resolve_strategy(backend, req, fam, T)
        anyfit(T) = any(g -> tilebytes(T, g.edge) <= budget, menu)
        for T in (Float32, Float64, ComplexF32, ComplexF64, Tbig, Tnone)
            @test resolve(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, T) ===
                (anyfit(T) ? Strided.BP_GROUPTILE : Strided.BP_ELEMENTWISE)
            @test resolve(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, T) === bp_expected_strategy(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, T)
            # the smallest edge alone decides, because tile bytes are monotone in the edge
            @test anyfit(T) == (tilebytes(T, minimum(edges)) <= budget)
            # never for the other families, whatever fits
            @test resolve(Strided.BP_GROUPTILE, Strided.BP_PAYLOAD, T) === Strided.BP_ELEMENTWISE
            @test resolve(Strided.BP_GROUPTILE, Strided.BP_COPY, T) === Strided.BP_ELEMENTWISE
        end
        # the two element types the byte math above singled out
        @test resolve(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, Tbig) === Strided.BP_GROUPTILE
        @test resolve(Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE, Tnone) === Strided.BP_ELEMENTWISE
        # End to end for the eligible one: on extents that tie between the two edges (every
        # extent a multiple of 32 or in the upper half of its last 32-block) a 4-byte element
        # type gets the larger edge, while this element type gets the 16 tile because the 32
        # one is over budget -- and the cooperative kernel runs bit-exactly on 64-byte
        # elements through it (NaN-poisoned destinations).
        shapes = [(64, 64), (20, 20), (50, 60), (96, 17)]
        plan = bp_runcase(AT, Float32, shapes, (2, 1), Strided.BP_GROUPTILE, Strided.BP_TRANSPOSE)
        @test plan.tile == (32, 32) == bp_expected_grouptile_tile(plan)
        srcs_cpu = [rand(Tbig, s...) for s in shapes]
        srcs = [AT(s) for s in srcs_cpu]
        dsts = [AT(fill(bigpoison, size(s, 2), size(s, 1))) for s in srcs_cpu]
        plan = Strided.plan_batched_permutedims(dsts, srcs, (2, 1); strategy = Strided.BP_GROUPTILE)
        @test plan.strategy === Strided.BP_GROUPTILE
        @test plan.family === Strided.BP_TRANSPOSE
        @test plan.tile == (16, 16) == bp_expected_grouptile_tile(plan)
        @test plan.totaltiles == 16 + 4 + 16 + 12
        out = Strided.batched_permutedims!(dsts, srcs, plan)
        @test out === dsts
        for (d, s) in zip(dsts, srcs_cpu)
            @test isequal(Array(d), permutedims(s, (2, 1)))
        end
        for (s, s0) in zip(srcs, srcs_cpu)
            @test isequal(Array(s), s0)
        end
        # the ineligible one falls back and still runs correctly
        srcs_cpu = [rand(Tnone, 9, 5)]
        srcs = [AT(s) for s in srcs_cpu]
        dsts = [AT(fill(ntuple(Returns(NaN), 16), 5, 9))]
        plan = Strided.plan_batched_permutedims(dsts, srcs, (2, 1); strategy = Strided.BP_GROUPTILE)
        @test plan.strategy === Strided.BP_ELEMENTWISE
        Strided.batched_permutedims!(dsts, srcs, plan)
        @test isequal(Array(dsts[1]), permutedims(srcs_cpu[1], (2, 1)))
    end

    @testset "plan.tile follows the exact per-tensor utilization rule" begin
        S = Strided.BP_GROUPTILE
        # Every case is asserted twice: against the rule restated in `bp_expected_grouptile_tile`
        # on the plan's own descriptors, and against a hand-computed edge. Hand arithmetic
        # (rank 2): a tensor `d1 x dc` schedules `cld(d1, e) * cld(dc, e) * e^2` slots on edge
        # `e`; the batch's edge is the one with fewer slots in total, 32 on a tie.
        function tileof(shapes, perm)
            plan = bp_runcase(AT, Float32, shapes, perm, S, Strided.BP_TRANSPOSE)
            @test plan.tile == bp_expected_grouptile_tile(plan)
            return plan
        end
        # all small (within 16 on both active axes), rank 2 and rank 3 with the
        # destination-fastest axis being source axis 3 (perm (3,1,2)) or 2 (perm (2,3,1))
        plan = tileof([(16, 16), (3, 16), (16, 1), (1, 1), (15, 9)], (2, 1))
        @test plan.tile == (16, 16)
        @test plan.totaltiles == 5
        plan = tileof([(16, 5, 16), (2, 3, 4), (16, 16, 16), (1, 7, 15)], (3, 1, 2))
        @test plan.tile == (16, 1, 16) && plan.dstseq[1] == 3
        plan = tileof([(16, 16, 5), (4, 2, 3), (16, 16, 16), (15, 1, 7)], (2, 3, 1))
        @test plan.tile == (16, 16, 1) && plan.dstseq[1] == 2
        # mixed: one 40x40 (32: 2x2 tiles = 4096 slots; 16: 3x3 = 2304) plus fifty 16x16
        # (32: 51200; 16: 12800) -- 55296 vs 15104 slots for 14400 elements: 16. A rule
        # working from the batch's per-axis maxima (40, 40) alone could not see this.
        shapes = Tuple{Int, Int}[(40, 40)]
        append!(shapes, [(16, 16) for _ in 1:50])
        plan = tileof(shapes, (2, 1))
        @test plan.tile == (16, 16)
        @test plan.totaltiles == 9 + 50
        # the same mix at rank 3, either placement of the destination-fastest axis
        plan = tileof(vcat([(40, 3, 40)], [(16, 2, 16) for _ in 1:50]), (3, 1, 2))
        @test plan.tile == (16, 1, 16) && plan.dstseq[1] == 3
        plan = tileof(vcat([(40, 40, 3)], [(16, 16, 2) for _ in 1:50]), (2, 3, 1))
        @test plan.tile == (16, 16, 1) && plan.dstseq[1] == 2
        # varied moderate-to-large: 64x64 (4096 / 4096), 65x33 (6144 / 3840), 16x16
        # (1024 / 256), 512x3 (16384 / 8192) -- 27648 vs 16384: 16
        plan = tileof([(64, 64), (65, 33), (16, 16), (512, 3)], (2, 1))
        @test plan.tile == (16, 16)
        # a single extent of 17 on an active axis: 1 tile of 32 (1024) vs 2x1 of 16 (512),
        # on top of small tensors -- 16 (a per-axis-maxima rule would have kept 32 here)
        plan = tileof([(17, 16), (16, 16), (1, 1)], (2, 1))
        @test plan.tile == (16, 16)
        plan = tileof([(16, 17), (16, 16)], (2, 1))
        @test plan.tile == (16, 16)
        # tie: every active extent is a multiple of 32 or lands in the upper half of its last
        # 32-block (17, 20, 50, 60, 96: `cld(d, 32) * 32 == cld(d, 16) * 16`), so both edges
        # idle exactly the same slots -- the tie goes to the larger edge
        plan = tileof([(20, 20), (50, 60), (64, 64), (96, 17)], (2, 1))
        @test plan.tile == (32, 32)
        # ... and a single lower-half extent anywhere breaks that tie toward 16: ten 64x64
        # (40960 slots either way) plus one 33x33 (4096 vs 2304)
        plan = tileof(vcat([(64, 64) for _ in 1:10], [(33, 33)]), (2, 1))
        @test plan.tile == (16, 16)
        # uniformly large, full tiles only: tie -> 32
        plan = tileof([(64, 64), (128, 96), (256, 32)], (2, 1))
        @test plan.tile == (32, 32)
        plan = tileof([(64, 5, 96), (128, 2, 32)], (3, 1, 2))
        @test plan.tile == (32, 1, 32) && plan.dstseq[1] == 3
        # empty tensors contribute neither tiles nor elements, so they never sway the choice
        plan = tileof([(0, 16), (16, 16), (16, 0), (40, 40)], (2, 1))
        @test plan.tile == (16, 16)
        plan = tileof([(0, 5), (64, 64), (7, 0)], (2, 1))
        @test plan.tile == (32, 32)
        # all empty: no tiles at all, the tie-break default
        plan = tileof([(0, 16), (16, 0)], (2, 1))
        @test plan.tile == (32, 32)
        @test plan.totaltiles == 0
        # the non-active axis multiplies both edges' slot counts alike: no effect
        plan = tileof([(16, 300, 16), (2, 1, 4)], (3, 1, 2))
        @test plan.tile == (16, 1, 16)
        plan = tileof([(64, 300, 64)], (3, 1, 2))
        @test plan.tile == (32, 1, 32)
    end

    @testset "bit-exact on a mixed batch through the 16x16 geometry (Float32, rank 2 and 3)" begin
        S = Strided.BP_GROUPTILE
        # Full 16x16 tiles, tiles partial on one or both axes, multi-tile tensors and
        # single-element ones, in one batch; the distinct shapes alone total 22528 slots on
        # 32 vs 13568 on 16, and the appended 16x16 tensors only widen that -- so this runs
        # on the 16 edge with both full and partial tiles.
        base = [(40, 40), (33, 17), (16, 16), (1, 16), (31, 2), (48, 20), (17, 33), (65, 1), (1, 1), (32, 32), (47, 49)]
        shapes = vcat(base, [(16, 16) for _ in 1:30])
        plan2 = bp_runcase(AT, Float32, shapes, (2, 1), S, Strided.BP_TRANSPOSE)
        @test plan2.tile == (16, 16) == bp_expected_grouptile_tile(plan2)
        @test plan2.totaltiles == sum(prod(cld.(s, 16)) for s in shapes)
        m = Iterators.cycle((1, 2, 3))
        shapes3a = [(i, mm, j) for ((i, j), mm) in zip(shapes, m)]
        plan3a = bp_runcase(AT, Float32, shapes3a, (3, 1, 2), S, Strided.BP_TRANSPOSE)
        @test plan3a.tile == (16, 1, 16) == bp_expected_grouptile_tile(plan3a) && plan3a.dstseq[1] == 3
        shapes3b = [(i, j, mm) for ((i, j), mm) in zip(shapes, m)]
        plan3b = bp_runcase(AT, Float32, shapes3b, (2, 3, 1), S, Strided.BP_TRANSPOSE)
        @test plan3b.tile == (16, 16, 1) == bp_expected_grouptile_tile(plan3b) && plan3b.dstseq[1] == 2
        # views with a guard band: one large offset block next to reversed/strided/tiny
        # views, each carved from its own parent, so per-tensor offsets and strides differ
        P = (90, 90)
        plan = bp_viewcase(AT, Float32, (2, 1), P, P, [
                ((3:42, 5:44), (2:41, 7:46)),               # 40 x 40 dense offset block
                ((81:-2:3, 5:21), (2:3:50, 7:46)),          # 40 x 17: reversed dst, stride-3 src
                ((3:18, 5:16), (13:-1:2, 7:22)),            # 16 x 12: reversed source axis 1
                ((60:75, 60:75), (60:75, 60:75)),           # 16 x 16, one full tile
                ((5:5, 1:33), (1:33, 5:5)),                 # 1 x 33
            ], S)
        @test plan.strategy === S
        @test plan.tile == (16, 16) == bp_expected_grouptile_tile(plan)
        P3 = (50, 50, 50)
        plan = bp_viewcase(AT, Float32, (3, 1, 2), P3, P3, [
                ((44:-1:5, 5:37, 2:4), (2:34, 5:7, 3:42)),  # 40 x 33 x 3, reversed dst axis 1
                ((5:20, 5:19, 2:4), (16:-1:2, 5:7, 3:18)),  # 16 x 15 x 3, reversed src axis 1
                ((1:1, 1:1, 3:19), (1:1, 3:19, 4:4)),       # 1 x 1 x 17
            ], S)
        @test plan.strategy === S
        @test plan.tile == (16, 1, 16) == bp_expected_grouptile_tile(plan)
    end

    @testset "bit-exact on partial tiles through the 32x32 geometry (Float32, rank 2 and 3)" begin
        S = Strided.BP_GROUPTILE
        # The 32 edge is only ever chosen on a tie, i.e. when every tensor's two active
        # extents are multiples of 32 or land in the upper half of a 32-block (a remainder
        # of 17..31) -- those are exactly the partial tiles the 32x32 kernel meets through
        # the planner, so its edge guards are exercised on them here, on every pair of such
        # extents.
        exts = (17, 20, 31, 32, 50, 63, 64, 96)
        grid = [(i, j) for i in exts for j in exts]
        plan2 = bp_runcase(AT, Float32, grid, (2, 1), S, Strided.BP_TRANSPOSE)
        @test plan2.tile == (32, 32) == bp_expected_grouptile_tile(plan2)
        @test plan2.totaltiles == sum(prod(cld.(s, 32)) for s in grid)
        m = Iterators.cycle((1, 2, 3))
        grid3a = [(i, mm, j) for ((i, j), mm) in zip(grid, m)]
        plan3a = bp_runcase(AT, Float32, grid3a, (3, 1, 2), S, Strided.BP_TRANSPOSE)
        @test plan3a.tile == (32, 1, 32) == bp_expected_grouptile_tile(plan3a) && plan3a.dstseq[1] == 3
        grid3b = [(i, j, mm) for ((i, j), mm) in zip(grid, m)]
        plan3b = bp_runcase(AT, Float32, grid3b, (2, 3, 1), S, Strided.BP_TRANSPOSE)
        @test plan3b.tile == (32, 32, 1) == bp_expected_grouptile_tile(plan3b) && plan3b.dstseq[1] == 2
        # views with a guard band: 50 x 60 destinations (partial 32-tiles on both axes),
        # offset, strided and reversed
        P = (200, 200)
        for (dr, sr) in (
                ((3:52, 5:64), (2:61, 7:56)),               # offset, dense
                ((101:-2:3, 5:64), (2:3:179, 7:56)),        # reversed stride-2 dst, stride-3 src
                ((52:-1:3, 5:64), (61:-1:2, 7:56)),         # both axis-1 reversed
            )
            plan = bp_viewcase(AT, Float32, (2, 1), P, P, [(dr, sr)], S)
            @test plan.strategy === S
            @test plan.tile == (32, 32) == bp_expected_grouptile_tile(plan)
        end
    end

    @testset "bit-exact through the 16x16 geometry (Float32, rank 2 and 3)" begin
        S = Strided.BP_GROUPTILE
        exts = (1, 2, 15, 16)
        # every extent pair around the small edge on both active axes
        grid = [(i, j) for i in exts for j in exts]
        plan2 = bp_runcase(AT, Float32, grid, (2, 1), S, Strided.BP_TRANSPOSE)
        @test plan2.tile == (16, 16)
        m = Iterators.cycle((1, 2, 3))
        grid3a = [(i, mm, j) for ((i, j), mm) in zip(grid, m)]
        plan3a = bp_runcase(AT, Float32, grid3a, (3, 1, 2), S, Strided.BP_TRANSPOSE)
        @test plan3a.tile == (16, 1, 16)
        grid3b = [(i, j, mm) for ((i, j), mm) in zip(grid, m)]
        plan3b = bp_runcase(AT, Float32, grid3b, (2, 3, 1), S, Strided.BP_TRANSPOSE)
        @test plan3b.tile == (16, 16, 1)
        # deep tile->tensor lookup with the small geometry: 320 single-tile tensors
        cycle = [(5, 7), (16, 16), (1, 16), (16, 1), (2, 2), (15, 16), (16, 3), (3, 15), (9, 9), (1, 1)]
        shapes = [cycle[mod1(i, length(cycle))] for i in 1:320]
        plan = bp_runcase(AT, Float32, shapes, (2, 1), S, Strided.BP_TRANSPOSE)
        @test plan.tile == (16, 16)
        @test plan.totaltiles == 320 && length(plan.prefix) == 321
        # offset / strided / negative-stride views with a guard band, all within 16 extents
        P = (60, 60)
        for (dr, sr) in (
                ((3:18, 5:16), (2:13, 7:22)),             # offset sub-blocks, dense: dst 16x12, src 12x16
                ((3:2:33, 5:16), (2:13, 7:22)),           # stride-2 destination axis 1
                ((3:18, 5:16), (2:3:35, 7:22)),           # stride-3 source axis 1
                ((18:-1:3, 5:16), (13:-1:2, 7:22)),       # both axis-1 reversed
                ((33:-2:3, 5:16), (35:-3:2, 7:22)),       # reversed and strided
            )
            plan = bp_viewcase(AT, Float32, (2, 1), P, P, [(dr, sr)], S)
            @test plan.strategy === S
            @test plan.tile == (16, 16)
        end
        P3 = (30, 30, 30)
        plan = bp_viewcase(AT, Float32, (3, 1, 2), P3, P3,
            [((20:-1:5, 5:19, 2:4), (2:16, 5:7, 3:18)), ((5:20, 5:19, 2:4), (16:-1:2, 5:7, 3:18))], S)
        @test plan.strategy === S
        @test plan.tile == (16, 1, 16)
        # plan reuse keeps the small geometry across calls with fresh arrays
        shapes = [(16, 16), (3, 16), (15, 9)]
        _, srcs1, dsts1, _ = bp_fixture(AT, Float32, shapes, (2, 1))
        plan = Strided.plan_batched_permutedims(dsts1, srcs1, (2, 1); strategy = S)
        @test plan.tile == (16, 16)
        for _ in 1:2
            srcs_cpu, srcs, dsts, refs = bp_fixture(AT, Float32, shapes, (2, 1))
            Strided.batched_permutedims!(dsts, srcs, plan)
            for (d, r) in zip(dsts, refs)
                @test bp_refequal(d, r)
            end
        end
    end

    @testset "executor rejects a tile edge that is not a menu entry or does not fit the element type" begin
        _, srcs, dsts, _ = bp_fixture(AT, Float32, [(16, 16), (3, 16)], (2, 1))
        plan = Strided.plan_batched_permutedims(dsts, srcs, (2, 1); strategy = Strided.BP_GROUPTILE)
        @test plan.strategy === Strided.BP_GROUPTILE
        P = typeof(plan)
        rebuild(tile) = P(plan.perm, plan.srcorder, plan.groups, plan.dstseq, plan.family, plan.strategy,
            tile, plan.srclabels, plan.dstlabels, plan.descs, plan.cpublocks, plan.prefix,
            plan.totaltiles, plan.totalelements, plan.deviceid, Base.RefValue{Any}(nothing))
        for edge in (8, 24, 64, 1)
            edge in edges && continue
            @test_throws ArgumentError Strided.batched_permutedims!(dsts, srcs, rebuild((edge, edge)))
        end
        # a menu edge on axis 1 but a different one on axis c is not a valid geometry either
        @test_throws ArgumentError Strided.batched_permutedims!(dsts, srcs, rebuild((16, 32)))
        @test_throws ArgumentError Strided.batched_permutedims!(dsts, srcs, rebuild((32, 16)))
        # A menu edge whose tile does not fit the plan's element type is refused before any
        # launch: the planner gives this 64-byte element type the 16 tile; forcing the 32
        # tile onto the same plan must not reach the kernel (its tile would be over budget).
        srcsb = [AT(rand(Tbig, 16, 16)), AT(rand(Tbig, 3, 16))]
        dstsb = [AT(fill(bigpoison, 16, 16)), AT(fill(bigpoison, 16, 3))]
        planb = Strided.plan_batched_permutedims(dstsb, srcsb, (2, 1); strategy = Strided.BP_GROUPTILE)
        @test planb.strategy === Strided.BP_GROUPTILE
        @test planb.tile == (16, 16)
        Pb = typeof(planb)
        badb = Pb(planb.perm, planb.srcorder, planb.groups, planb.dstseq, planb.family, planb.strategy,
            (EDGE, EDGE), planb.srclabels, planb.dstlabels, planb.descs, planb.cpublocks, planb.prefix,
            planb.totaltiles, planb.totalelements, planb.deviceid, Base.RefValue{Any}(nothing))
        @test_throws ArgumentError Strided.batched_permutedims!(dstsb, srcsb, badb)
        @test all(x -> isequal(x, bigpoison), Array(dstsb[1]))   # nothing was written
        # the untouched plan still runs, bit-exactly
        Strided.batched_permutedims!(dstsb, srcsb, planb)
        @test isequal(Array(dstsb[1]), permutedims(Array(srcsb[1]), (2, 1)))
        @test isequal(Array(dstsb[2]), permutedims(Array(srcsb[2]), (2, 1)))
    end
end

# ---------- batched out-of-place permutation: per-tensor alpha/beta on the GPU ----------
#
# `dsts[b] = alpha[b] * permutedims(srcs[b], perm) + beta[b] * dsts[b]` on every strategy.
# Unlike the plain copy this computes, and a GPU may fuse the multiply-add where the CPU
# reference does not, so `isequal` is only a sound oracle on exactly-representable data:
# small integer-valued entries and small integer / half-integer coefficients, for which IEEE
# arithmetic is bit-identical whatever the evaluation order or fusion. A mismatch here is a
# real finding, never something to relax to `isapprox`. Kept to a few strategy x type x rank
# combinations on purpose: each scaled (strategy, T, rank, edge) is one more compiled kernel.

bp_exactdata(::Type{T}, dims) where {T <: Real} = T.(rand(-8:8, dims...))
bp_exactdata(::Type{T}, dims) where {T <: Complex} = T.(complex.(rand(-8:8, dims...), rand(-8:8, dims...)))
bp_exactcoeffs(::Type{T}) where {T <: Real} = T[-3, -1, 0, 0.5, 1, 2, 5]
bp_exactcoeffs(::Type{T}) where {T <: Complex} = T[-3, -1, 0, 0.5, 1, 2, 5, 1 - 2im]
bp_scaledref(a, s, b, d0, perm) = iszero(b) ? a .* permutedims(s, perm) : a .* permutedims(s, perm) .+ b .* d0
bp_dstshape(s, perm) = ntuple(j -> size(s, perm[j]), length(perm))

# alpha/beta cycling through the exact set with a shift: tensor 1 has beta == 0 (NaN-poisoned
# destination, must come out clean), tensor 3 has alpha == 0, later tensors have beta != 0
function bp_scaledcoeffs(::Type{T}, B::Int) where {T}
    cs = bp_exactcoeffs(T)
    n = length(cs)
    return [cs[mod1(b, n)] for b in 1:B], [cs[mod1(b + 2, n)] for b in 1:B]
end

# plan with `strategy`, run with coefficients, check the return value, bit-exactness against
# the CPU reference, and unmodified sources; returns the plan for strategy/tile assertions
function bp_scaledcase(AT, T, shapes, perm, strategy; coeffs = bp_scaledcoeffs(T, length(shapes)))
    alpha, beta = coeffs
    srcs_cpu = [bp_exactdata(T, s) for s in shapes]
    dsts_cpu = [iszero(beta[b]) ? fill(T(NaN), bp_dstshape(s, perm)) : bp_exactdata(T, bp_dstshape(s, perm))
                for (b, s) in enumerate(srcs_cpu)]
    srcs = [AT(s) for s in srcs_cpu]
    dsts = [AT(d) for d in dsts_cpu]
    plan = Strided.plan_batched_permutedims(dsts, srcs, perm; strategy)
    out = Strided.batched_permutedims!(dsts, srcs, plan; alpha, beta)
    @test out === dsts
    for b in eachindex(dsts)
        @test bp_refequal(dsts[b], bp_scaledref(T(alpha[b]), srcs_cpu[b], T(beta[b]), dsts_cpu[b], perm))
    end
    for (s, s0) in zip(srcs, srcs_cpu)
        @test bp_refequal(s, s0)
    end
    return plan
end

# `bp_viewcase` with coefficients: each view carved from its own parent pair, the destination
# parent NaN-poisoned where beta == 0 and exact data elsewhere, and the whole parent compared
# (scaled bits inside the view, the original bits outside it)
function bp_scaled_viewcase(AT, T, perm, dparentsize, sparentsize, cases, strategy, alpha, beta)
    dphs = [iszero(b) ? fill(T(NaN), dparentsize...) : bp_exactdata(T, dparentsize) for b in beta]
    sphs = [bp_exactdata(T, sparentsize) for _ in cases]
    dps = [AT(p) for p in dphs]
    sps = [AT(p) for p in sphs]
    dvs = [view(StridedView(dp), dr...) for (dp, (dr, _)) in zip(dps, cases)]
    svs = [view(StridedView(sp), sr...) for (sp, (_, sr)) in zip(sps, cases)]
    plan = Strided.plan_batched_permutedims(dvs, svs, perm; strategy)
    out = Strided.batched_permutedims!(dvs, svs, plan; alpha, beta)
    @test out === dvs
    for (dp, sp, dph, sph, (dr, sr), a, b) in zip(dps, sps, dphs, sphs, cases, alpha, beta)
        expected = copy(dph)
        view(expected, dr...) .= bp_scaledref(T(a), view(sph, sr...), T(b), view(dph, dr...), perm)
        @test isequal(Array(dp), expected)
        @test isequal(Array(sp), sph)
    end
    return plan
end

@testset "batched_permutedims! GPU alpha/beta ($AT)" for AT in ATs
    Ext = Base.get_extension(Strided, :StridedGPUArraysExt)

    @testset "every strategy, T=$T" for T in (Float32, ComplexF64)
        # transpose family; on this mix BP_GROUPTILE picks the 16 edge
        shapes = [(67, 51), (33, 32), (1, 64), (31, 2), (32, 32), (16, 16), (50, 3), (5, 70)]
        for strategy in (Strided.BP_ELEMENTWISE, Strided.BP_THREADTILE, Strided.BP_GROUPTILE)
            plan = bp_scaledcase(AT, T, shapes, (2, 1), strategy)
            @test plan.strategy === strategy
            strategy === Strided.BP_GROUPTILE && @test plan.tile == (16, 16)
        end
    end

    @testset "BP_GROUPTILE 32 edge, rank 3, and the shared elementwise kernel on the payload family" begin
        # tie extents (multiples of 32 or upper half of a 32-block) keep the 32 edge; partial tiles
        plan = bp_scaledcase(AT, Float32, [(64, 64), (20, 20), (50, 60), (96, 17)], (2, 1), Strided.BP_GROUPTILE)
        @test plan.strategy === Strided.BP_GROUPTILE && plan.tile == (32, 32)
        plan = bp_scaledcase(AT, Float32, [(33, 2, 31), (1, 32, 33), (2, 1, 1), (31, 33, 2), (16, 3, 16)], (3, 1, 2), Strided.BP_GROUPTILE)
        @test plan.strategy === Strided.BP_GROUPTILE && plan.tile[1] == plan.tile[3] != 1 && plan.dstseq[1] == 3
        plan = bp_scaledcase(AT, Float32, [(5, 6, 7), (2, 33, 31), (1, 1, 32), (32, 2, 1)], (1, 3, 2), Strided.BP_THREADTILE)
        @test plan.family === Strided.BP_PAYLOAD && plan.strategy === Strided.BP_THREADTILE
        # one dominant tensor plus many tiny ones: the coefficient lookup follows the tensor
        # lookup across many prefix boundaries
        shapes = Tuple{Int, Int}[(129, 97)]
        append!(shapes, [(3, 2) for _ in 1:20])
        append!(shapes, [(1, 1) for _ in 1:5])
        bp_scaledcase(AT, Float32, shapes, (2, 1), Strided.BP_ELEMENTWISE)
        bp_scaledcase(AT, Float32, shapes, (2, 1), Strided.BP_GROUPTILE)
    end

    @testset "offset / strided / negative-stride views with guard band, strategy=$strategy" for
            strategy in (Strided.BP_ELEMENTWISE, Strided.BP_GROUPTILE)
        P = (210, 210)
        plan = bp_scaled_viewcase(AT, Float32, (2, 1), P, P, [
                ((3:42, 5:71), (2:68, 7:46)),              # offset sub-blocks, dense
                ((81:-2:3, 5:71), (2:3:200, 7:46)),        # reversed stride-2 dst, stride-3 src
                ((42:-1:3, 5:71), (68:-1:2, 7:46)),        # both axis-1 reversed
                ((100:139, 7:73), (50:116, 150:189)),      # far offsets
            ], strategy, Float32[2, -1, 0.5, 5], Float32[0, 1, -3, 0])
        @test plan.family === Strided.BP_TRANSPOSE && plan.strategy === strategy
        # the same through the 16 edge (all extents within 16)
        P = (60, 60)
        plan = bp_scaled_viewcase(AT, Float32, (2, 1), P, P, [
                ((3:18, 5:16), (2:13, 7:22)),
                ((33:-2:3, 5:16), (35:-3:2, 7:22)),
                ((18:-1:3, 5:16), (13:-1:2, 7:22)),
            ], strategy, Float32[-3, 1, 0.5], Float32[2, 0, 1])
        @test plan.strategy === strategy
        strategy === Strided.BP_GROUPTILE && @test plan.tile == (16, 16)
    end

    @testset "omitted alpha/beta is exactly ones/zeros; coefficients never touch the binding cache" begin
        T = Float32
        shapes = [(33, 17), (5, 40), (16, 16), (2, 65)]
        perm = (2, 1)
        B = length(shapes)
        srcs_cpu = [bp_exactdata(T, s) for s in shapes]
        d0 = [bp_exactdata(T, bp_dstshape(s, perm)) for s in srcs_cpu]
        srcs = [AT(s) for s in srcs_cpu]
        for strategy in (Strided.BP_ELEMENTWISE, Strided.BP_GROUPTILE)
            dsts = [AT(d) for d in d0]
            plan = Strided.plan_batched_permutedims(dsts, srcs, perm; strategy)
            runwith(a, b) = (copyto!.(dsts, d0); Strided.batched_permutedims!(dsts, srcs, plan; alpha = a, beta = b); map(Array, dsts))
            plain = runwith(nothing, nothing)
            b1 = plan.devcache[]
            @test b1 isa Ext._BatchedGPUBinding
            for (d, s) in zip(plain, srcs_cpu)
                @test isequal(d, permutedims(s, perm))
            end
            @test all(map(isequal, plain, runwith(ones(T, B), zeros(T, B))))
            alpha = T[2, -1, 0.5, 5]
            beta = T[1, 0, -3, 2]
            @test all(map(isequal, runwith(alpha, nothing), runwith(alpha, zeros(T, B))))
            @test all(map(isequal, runwith(nothing, beta), runwith(ones(T, B), beta)))
            scaled = runwith(alpha, beta)
            for b in 1:B
                @test isequal(scaled[b], bp_scaledref(alpha[b], srcs_cpu[b], beta[b], d0[b], perm))
            end
            # same arrays, different (or no) coefficients: the cached binding is reused as-is
            @test plan.devcache[] === b1
            runwith(reverse(alpha), reverse(beta))
            @test plan.devcache[] === b1
        end
    end

    @testset "validation happens before any write" begin
        T = Float32
        srcs = [AT(bp_exactdata(T, (3, 4))), AT(bp_exactdata(T, (5, 2)))]
        dsts = [AT(fill(T(NaN), 4, 3)), AT(fill(T(NaN), 2, 5))]
        plan = Strided.plan_batched_permutedims(dsts, srcs, (2, 1))
        @test_throws DimensionMismatch Strided.batched_permutedims!(dsts, srcs, plan; alpha = ones(T, 3))
        @test_throws DimensionMismatch Strided.batched_permutedims!(dsts, srcs, plan; beta = zeros(T, 1))
        @test_throws InexactError Strided.batched_permutedims!(dsts, srcs, plan; alpha = [1 + 2im, 1])
        @test all(d -> all(isnan, Array(d)), dsts)
        @test plan.devcache[] === nothing   # nothing was uploaded either
    end
end
