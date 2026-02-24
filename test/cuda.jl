for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
    @testset "Copy with CuStridedView: $T" begin
        m1 = 32
        m2 = 16
        A1 = CUDA.randn(T, (m1, m2))
        A2 = similar(A1)
        A1c = copy(A1)
        A2c = copy(A2)
        B1 = StridedView(A1c)
        B2 = StridedView(A2c)
        @test copy!(A2, A1) == copy!(B2, B1)
        @test copy!(transpose(A2), transpose(A1)) == copy!(transpose(B2), transpose(B1))
        if T <: Complex
            @test_broken copy!(transpose(A2), adjoint(A1)) == copy!(transpose(B2), adjoint(B1))
            @test_broken copy!(adjoint(A2), adjoint(A1)) == copy!(adjoint(B2), adjoint(B1))
            @test_broken copy!(A2, conj(A1)) == copy!(B2, conj(B1))
            @test_broken copy!(conj(A2), conj(A1)) == copy!(conj(B2), conj(B1))
        else
            @test copy!(transpose(A2), adjoint(A1)) == copy!(transpose(B2), adjoint(B1))
            @test copy!(adjoint(A2), adjoint(A1)) == copy!(adjoint(B2), adjoint(B1))
            @test copy!(A2, conj(A1)) == copy!(B2, conj(B1))
            @test copy!(conj(A2), conj(A1)) == copy!(conj(B2), conj(B1))
        end
    end
end
