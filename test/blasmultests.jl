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

    # `β = 0` overwrites `C` and ignores `NaN`s, also when the product can not be computed by BLAS
    for T in (Float64, ComplexF64), op in (identity, conj), α in (zero(T), one(T))
        A2 = StridedView(rand(T, 2, 0))
        B2 = StridedView(rand(T, 0, 2))
        C2 = StridedView(fill(T(NaN), 2, 2))
        @test iszero(mul!(C2, op(A2), B2, α, zero(T)))
    end
    for T in (Float64, ComplexF64), op in (identity, conj)
        A2 = StridedView(rand(T, 2, 3))
        B2 = StridedView(rand(T, 3, 2))
        C2 = StridedView(fill(T(NaN), 2, 2))
        @test iszero(mul!(C2, op(A2), B2, zero(T), zero(T)))
    end
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

@testset "BLAS layouts with size 1 dimensions" begin
    for T in (Float64, ComplexF64)
        m, n, k = 5, 4, 3
        M = randn(T, m, n)
        # a row and a column, with the stride of their size 1 dimension changed
        row = StridedView(M)[2:2, :] # strides (1, m)
        @test Strided.isblasmatrix(row)
        row2 = Strided.StridedViews.unsafe_stridedview(row, (1, n), (m, m))
        @test row2 == row
        @test Strided.isblasmatrix(row2)
        col = StridedView(M)[1:2:5, 2:2] # strides (2, 2m), only usable as 'T'
        @test Strided.isblasmatrix(col)
        @test Strided.getblasmatrix(col)[2] == 'T'
        @test strides(first(Strided.getblasmatrix(col))) == (1, 2)
        # negative strides and overlapping columns can not be passed to BLAS
        neg = StridedView(M)[:, end:-1:1]
        @test !Strided.isblasmatrix(neg)
        ovl = Strided.StridedViews.unsafe_stridedview(StridedView(M), (m, 3), (1, 2))
        @test !Strided.isblasmatrix(ovl)
        for A in (row, row2, col, neg, ovl), op in (identity, conj, transpose, adjoint)
            Aop = op(A)
            B = StridedView(randn(T, size(Aop, 2), k))
            C = StridedView(zeros(T, size(Aop, 1), k))
            α, β = randn(T), zero(T)
            @test mul!(C, Aop, B, α, β) ≈ α * Matrix(Aop) * Matrix(B)
            # and as output, transposed if needed
            C2 = StridedView(zeros(T, k, size(Aop, 1)))
            @test mul!(C2, transpose(B), transpose(Aop), α, β) ≈ α * transpose(Matrix(B)) * transpose(Matrix(Aop))
        end
        for C in (row2, col, Strided.StridedViews.unsafe_stridedview(StridedView(zeros(T, 3m)), (m, 1), (3, 1)))
            C0 = Matrix(C)
            A = StridedView(randn(T, size(C, 1), k))
            B = StridedView(randn(T, k, size(C, 2)))
            α, β = randn(T), randn(T)
            @test mul!(C, A, B, α, β) ≈ α * Matrix(A) * Matrix(B) + β * C0
            C0 = Matrix(C)
            @test mul!(conj(C), A, B, α, β) ≈ α * Matrix(A) * Matrix(B) + β * conj(C0)
        end
    end
end
