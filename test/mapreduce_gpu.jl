using Test, Strided, StridedViews, JLArrays

@testset "GPU map! via StridedView" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = JLArray(rand(T, 8, 6))
        B = similar(A)
        map!(x -> 2x, StridedView(B), StridedView(A))
        @test Array(B) ≈ 2 .* Array(A)
    end
end

@testset "GPU mapreducedim! — sum over dim 1" begin
    for T in (Float32, Float64)
        A = JLArray(rand(T, 8, 6))
        B = JLArray(zeros(T, 1, 6))
        sum!(StridedView(B), StridedView(A))
        @test Array(B) ≈ sum(Array(A); dims = 1)
    end
end

@testset "GPU mapreducedim! — sum over dim 2" begin
    for T in (Float32, Float64)
        A = JLArray(rand(T, 8, 6))
        B = JLArray(zeros(T, 8, 1))
        sum!(StridedView(B), StridedView(A))
        @test Array(B) ≈ sum(Array(A); dims = 2)
    end
end

@testset "GPU map! with conj/adjoint StridedView" begin
    for T in (ComplexF32, ComplexF64)
        A = JLArray(rand(T, 4, 4))
        B = JLArray(zeros(T, 4, 4))
        copy!(adjoint(StridedView(B)), StridedView(A))
        @test Array(B) ≈ adjoint(Array(A))
    end
end

@testset "GPU mapreduce — full scalar reduction" begin
    for T in (Float32, Float64)
        A = JLArray(rand(T, 8, 6))
        result = sum(StridedView(A))
        @test result isa T
        @test result ≈ sum(Array(A))
    end
end

# ---- nontrivial strides and offsets ----

@testset "GPU map! — stride-2 input (every other row)" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = JLArray(rand(T, 8, 6))
        B = JLArray(zeros(T, 4, 6))
        src = StridedView(A)[1:2:8, :]   # stride 2 in dim 1
        map!(x -> 2x, StridedView(B), src)
        @test Array(B) ≈ 2 .* Array(A)[1:2:8, :]
    end
end

@testset "GPU map! — stride-2 output (every other row)" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = JLArray(rand(T, 4, 6))
        B = JLArray(zeros(T, 8, 6))
        dst = StridedView(B)[1:2:8, :]   # stride 2 in dim 1
        map!(identity, dst, StridedView(A))
        @test Array(B)[1:2:8, :] ≈ Array(A)
        @test all(iszero, Array(B)[2:2:8, :])  # untouched rows stay zero
    end
end

@testset "GPU map! — subview with nonzero offset" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = JLArray(rand(T, 8, 6))
        B = JLArray(zeros(T, 5, 4))
        src = StridedView(A)[2:6, 3:6]   # offset = 1 row + 2 cols
        map!(x -> x + 1, StridedView(B), src)
        @test Array(B) ≈ Array(A)[2:6, 3:6] .+ 1
    end
end

@testset "GPU map! — permuted (transposed) strides" begin
    for T in (Float32, Float64, ComplexF32, ComplexF64)
        A = JLArray(rand(T, 6, 8))
        B = JLArray(zeros(T, 8, 6))
        src = permutedims(StridedView(A), (2, 1))   # strides (8,1) → (1,6) after permute: 8×6 view
        map!(identity, StridedView(B), src)
        @test Array(B) ≈ permutedims(Array(A), (2, 1))
    end
end

@testset "GPU sum over dim 1 — stride-2 input" begin
    for T in (Float32, Float64)
        A = JLArray(rand(T, 8, 6))
        B = JLArray(zeros(T, 1, 6))
        src = StridedView(A)[1:2:8, :]   # 4×6 with stride 2
        sum!(StridedView(B), src)
        @test Array(B) ≈ sum(Array(A)[1:2:8, :]; dims = 1)
    end
end

@testset "GPU sum over dim 2 — subview with offset" begin
    for T in (Float32, Float64)
        A = JLArray(rand(T, 8, 6))
        B = JLArray(zeros(T, 5, 1))
        src = StridedView(A)[2:6, 2:5]   # 5×4, offset = 1 row + 1 col
        sum!(StridedView(B), src)
        @test Array(B) ≈ sum(Array(A)[2:6, 2:5]; dims = 2)
    end
end

@testset "GPU full scalar reduction — stride-2 and offset subview" begin
    for T in (Float32, Float64)
        A = JLArray(rand(T, 8, 6))
        r1 = sum(StridedView(A)[1:2:8, :])    # stride-2
        @test r1 ≈ sum(Array(A)[1:2:8, :])
        r2 = sum(StridedView(A)[3:6, 2:5])    # offset subview
        @test r2 ≈ sum(Array(A)[3:6, 2:5])
    end
end
