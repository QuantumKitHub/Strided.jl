# Parameterized mapreduce / map! tests.
# Iterates over both Array and JLArray backends internally.

backends = [("Array", identity), ("JLArray", JLArray)]

for (backend_name, make_arr) in backends
    @testset "$backend_name: map! via StridedView" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            A = make_arr(rand(T, 8, 6))
            B = similar(A)
            map!(x -> 2x, StridedView(B), StridedView(A))
            @test Array(StridedView(B)) ≈ 2 .* Array(StridedView(A))
        end
    end

    @testset "$backend_name: mapreducedim! — sum over dim 1" begin
        for T in (Float32, Float64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 1, 6))
            sum!(StridedView(B), StridedView(A))
            @test Array(StridedView(B)) ≈ sum(data; dims = 1)
        end
    end

    @testset "$backend_name: mapreducedim! — sum over dim 2" begin
        for T in (Float32, Float64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 8, 1))
            sum!(StridedView(B), StridedView(A))
            @test Array(StridedView(B)) ≈ sum(data; dims = 2)
        end
    end

    @testset "$backend_name: map! with conj/adjoint StridedView" begin
        for T in (ComplexF32, ComplexF64)
            data = rand(T, 4, 4)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 4, 4))
            copy!(adjoint(StridedView(B)), StridedView(A))
            @test Array(StridedView(B)) ≈ adjoint(data)
        end
    end

    @testset "$backend_name: mapreduce — full scalar reduction" begin
        for T in (Float32, Float64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            result = sum(StridedView(A))
            @test result isa T
            @test result ≈ sum(data)
        end
    end

    # Only meaningful for GPU backends: mixing CPU and GPU inputs must not silently
    # use the GPU dispatch path.
    if make_arr !== identity
        @testset "$backend_name: dispatch requires all inputs on GPU" begin
            A_gpu = make_arr(rand(Float32, 4, 4))
            A_cpu = Array(StridedView(A_gpu))
            B_gpu = make_arr(zeros(Float32, 4, 4))
            @test_throws Exception map!(+, StridedView(B_gpu), StridedView(A_gpu), StridedView(A_cpu))
        end
    end

    @testset "$backend_name: map! — stride-2 input (every other row)" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 4, 6))
            src = StridedView(A)[1:2:8, :]
            map!(x -> 2x, StridedView(B), src)
            @test Array(StridedView(B)) ≈ 2 .* data[1:2:8, :]
        end
    end

    @testset "$backend_name: map! — stride-2 output (every other row)" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            data = rand(T, 4, 6)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 8, 6))
            dst = StridedView(B)[1:2:8, :]
            map!(identity, dst, StridedView(A))
            B_cpu = Array(StridedView(B))
            @test B_cpu[1:2:8, :] ≈ data
            @test all(iszero, B_cpu[2:2:8, :])
        end
    end

    @testset "$backend_name: map! — subview with nonzero offset" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 5, 4))
            src = StridedView(A)[2:6, 3:6]
            map!(x -> x + 1, StridedView(B), src)
            @test Array(StridedView(B)) ≈ data[2:6, 3:6] .+ 1
        end
    end

    @testset "$backend_name: map! — permuted (transposed) strides" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            data = rand(T, 6, 8)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 8, 6))
            src = permutedims(StridedView(A), (2, 1))
            map!(identity, StridedView(B), src)
            @test Array(StridedView(B)) ≈ permutedims(data, (2, 1))
        end
    end

    @testset "$backend_name: sum over dim 1 — stride-2 input" begin
        for T in (Float32, Float64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 1, 6))
            src = StridedView(A)[1:2:8, :]
            sum!(StridedView(B), src)
            @test Array(StridedView(B)) ≈ sum(data[1:2:8, :]; dims = 1)
        end
    end

    @testset "$backend_name: sum over dim 2 — subview with offset" begin
        for T in (Float32, Float64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            B = make_arr(zeros(T, 5, 1))
            src = StridedView(A)[2:6, 2:5]
            sum!(StridedView(B), src)
            @test Array(StridedView(B)) ≈ sum(data[2:6, 2:5]; dims = 2)
        end
    end

    @testset "$backend_name: full scalar reduction — stride-2 and offset subview" begin
        for T in (Float32, Float64)
            data = rand(T, 8, 6)
            A = make_arr(copy(data))
            r1 = sum(StridedView(A)[1:2:8, :])
            @test r1 ≈ sum(data[1:2:8, :])
            r2 = sum(StridedView(A)[3:6, 2:5])
            @test r2 ≈ sum(data[3:6, 2:5])
        end
    end
end
