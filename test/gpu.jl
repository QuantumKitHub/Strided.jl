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
CUDACore.functional() && push!(ATs, CuArray)
AMDGPU.functional() && push!(ATs, ROCArray)
Metal.functional() && push!(ATs, MtlArray)

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
            @test compare((x, y, z) -> map((a, b, c) -> sin(a) + b / exp(-abs(c)), x, y, z), AT, A1, A2, A3)
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
        @test compare((x, y, z) -> map((a, b, c) -> sin(a) + b / exp(-abs(c)), x, y, z), AT, A1, A2, A3)
        @test compare((x, y) -> mul!(x, 1, y), AT, A1, A2)
        @test compare((x, y) -> mul!(x, y, 1), AT, A1, A2)
    end
end

@testset "broadcasting ($AT)" for AT in ATs
    for T in (Float32, ComplexF32)
        A0 = StridedView(rand(T, ()))
        A1 = StridedView(rand(T, (10,)))
        A2 = permutedims(StridedView(rand(T, (10, 10))), randperm(2))
        A3 = permutedims(StridedView(rand(T, (10, 10, 10))), randperm(3))
        A4 = StridedView(rand(T, (2, 0)))

        @test compare((x, y) -> x .+ sin.(y .- 3), AT, A1, A2)
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
        @test compare(x -> mapreduce(sin, +, x; dims = (1, 3, 5)), AT, A1)
        @test compare(x -> sum(x; dims = (1, 3, 5)), AT, permutedims(A1, randperm(N)))
        @test compare(x -> mapreduce(sin, +, x; dims = (1, 3, 5)), AT, permutedims(A1, randperm(N)))

        A2 = sreshape(StridedView(rand(T, ntuple(Returns(sz), 3))), (sz, 1, 1, sz, sz, 1))

        @test compare((x, y) -> Strided._mapreducedim!(sin, +, identity, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
        @test compare((x, y) -> Strided._mapreducedim!(sin, +, Returns(0), ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
        @test compare((x, y) -> Strided._mapreducedim!(sin, +, conj, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)

        β = rand(T)
        @test compare((x, y) -> Strided._mapreducedim!(sin, +, a -> β, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
        @test compare((x, y) -> Strided._mapreducedim!(sin, +, a -> β * a, ntuple(Returns(sz), N), (x, y)), AT, A1, A2)
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
