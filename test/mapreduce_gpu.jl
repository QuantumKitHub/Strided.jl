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
