for T1 in (Float32, Float64, Complex{Float32}, Complex{Float64})
    for T2 in (Float32, Float64, Complex{Float32}, Complex{Float64})
        @testset "Matrix multiplication with StridedView: $T1 times $T2" begin
            d = 103
            A1 = rand(T1, (d, d))
            A2 = rand(T2, (d, d))
            T3 = promote_type(T1, T2)
            A3 = rand(T3, (d, d))
            A4 = rand(T3, (d, d))
            B1 = StridedView(A1)
            B2 = StridedView(A2)
            B3 = StridedView(A3)
            B4 = StridedView(A4)

            for op1 in (identity, conj, transpose, adjoint)
                @test op1(A1) == op1(B1)
                for op2 in (identity, conj, transpose, adjoint)
                    @test op1(A1) * op2(A2) ≈ op1(B1) * op2(B2)
                    for op3 in (identity, conj, transpose, adjoint)
                        α = randn(T3)
                        β = randn(T3)
                        copy!(A3, A4)
                        mul!(op3(B3), op1(B1), op2(B2), α, β)
                        @test B3 ≈ op3(β) * A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                    end
                end
            end
        end

        @testset "Outer product with StridedView: $T1 times $T2" begin
            d = 10
            A1 = rand(T1, (d, 1))
            A2 = rand(T2, (1, d))
            T3 = promote_type(T1, T2)
            A3 = rand(T3, (d, d))
            A4 = rand(T3, (d, d))
            B1 = StridedView(A1)
            B2 = StridedView(A2)
            B3 = StridedView(A3)
            B4 = StridedView(A4)
            α = randn(T3)
            β = randn(T3)
            for op1 in (identity, conj)
                @test op1(A1) == op1(B1)
                for op2 in (identity, conj)
                    @test op1(A1) * op2(A2) ≈ op1(B1) * op2(B2)
                    for op3 in (identity, conj, transpose, adjoint)
                        α = randn(T3)
                        β = randn(T3)
                        copy!(A3, A4)
                        mul!(op3(B3), op1(B1), op2(B2), α, β)
                        @test B3 ≈ op3(β) * A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                    end
                end
            end
        end

        @testset "Inner product with StridedView: $T1 times $T2" begin
            d = 10
            A1 = rand(T1, (1, d))
            A2 = rand(T2, (d, 1))
            T3 = promote_type(T1, T2)
            A3 = rand(T3, (1, 1))
            A4 = rand(T3, (1, 1))
            B1 = StridedView(A1)
            B2 = StridedView(A2)
            B3 = StridedView(A3)
            B4 = StridedView(A4)
            α = randn(T3)
            β = randn(T3)
            for op1 in (identity, conj)
                @test op1(A1) == op1(B1)
                for op2 in (identity, conj)
                    @test op1(A1) * op2(A2) ≈ op1(B1) * op2(B2)
                    for op3 in (identity, conj, transpose, adjoint)
                        α = randn(T3)
                        β = randn(T3)
                        copy!(A3, A4)
                        mul!(op3(B3), op1(B1), op2(B2), α, β)
                        @test B3 ≈ op3(β) * A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                    end
                end
            end
        end
    end
end

@testset "Matrix multiplication with length 0" begin
    A = rand(2, 0)
    B = rand(0, 2)
    C = rand(2, 2)
    α = rand()
    β = rand()
    A1 = StridedView(copy(A))
    B1 = StridedView(copy(B))
    C1 = StridedView(copy(C))
    @test mul!(C, A, B, α, β) ≈ mul!(C1, A1, B1, α, β)
end

# small enough to never be split over threads, such that no tasks need to be spawned
blasmul_allocations(C, A, B, α, β) = @allocated mul!(C, A, B, α, β)
@testset "Matrix multiplication with StridedView does not allocate: $T" for T in (Float64, ComplexF64)
    d = 8
    A = StridedView(rand(T, (d, d)))
    B = StridedView(rand(T, (d, d)))
    C = StridedView(rand(T, (d, d)))
    α = rand(T)
    β = rand(T)
    for op1 in (identity, conj, transpose, adjoint), op2 in (identity, conj, transpose, adjoint)
        blasmul_allocations(C, op1(A), op2(B), α, β) # compile
        @test blasmul_allocations(C, op1(A), op2(B), α, β) == 0
    end
end
