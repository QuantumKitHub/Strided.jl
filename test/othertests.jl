@testset "in-place matrix operations" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A1 = randn(T, (1000, 1000))
        A2 = similar(A1)
        A1c = copy(A1)
        A2c = copy(A2)
        B1 = StridedView(A1c)
        B2 = StridedView(A2c)

        @test conj!(A1) == conj!(B1)
        @test adjoint!(A2, A1) == adjoint!(B2, B1)
        @test transpose!(A2, A1) == transpose!(B2, B1)
        @test permutedims!(A2, A1, (2, 1)) == permutedims!(B2, B1, (2, 1))
    end
end

@testset "map, scale!, axpy! and axpby! with StridedView" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        @testset for N in 2:6
            dims = ntuple(n -> div(60, N), N)
            R1, R2, R3 = rand(T, dims), rand(T, dims), rand(T, dims)
            B1 = permutedims(StridedView(R1), randperm(N))
            B2 = permutedims(StridedView(R2), randperm(N))
            B3 = permutedims(StridedView(R3), randperm(N))
            A1 = convert(Array, B1)
            A2 = convert(Array{T}, B2) # test different converts
            A3 = convert(Array{T, N}, B3)
            C1 = deepcopy(B1)

            @test rmul!(B1, 1 // 2) ≈ rmul!(A1, 1 // 2)
            @test lmul!(1 // 3, B2) ≈ lmul!(1 // 3, A2)
            @test axpy!(1 // 3, B1, B2) ≈ axpy!(1 // 3, A1, A2)
            @test axpy!(1, B2, B3) ≈ axpy!(1, A2, A3)
            @test axpby!(1 // 3, B1, 1 // 2, B3) ≈ axpby!(1 // 3, A1, 1 // 2, A3)
            @test axpby!(1, B2, 1, B1) ≈ axpby!(1, A2, 1, A1)
            @test map((x, y, z) -> sin(x) + y / exp(-abs(z)), B1, B2, B3) ≈
                map((x, y, z) -> sin(x) + y / exp(-abs(z)), A1, A2, A3)
            @test map((x, y, z) -> sin(x) + y / exp(-abs(z)), B1, B2, B3) isa StridedView
            @test map((x, y, z) -> sin(x) + y / exp(-abs(z)), B1, A2, B3) isa Array
            @test mul!(B1, 1, B2) ≈ mul!(A1, 1, A2)
            @test mul!(B1, B2, 1) ≈ mul!(A1, A2, 1)
        end
    end
end

@testset "broadcast with StridedView" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        R1, R2, R3 = rand(T, (10,)), rand(T, (10, 10)), rand(T, (10, 10, 10))
        B1 = StridedView(R1)
        B2 = permutedims(StridedView(R2), randperm(2))
        B3 = permutedims(StridedView(R3), randperm(3))
        A1 = convert(Array, B1)
        A2 = convert(Array{T}, B2)
        A3 = convert(Array{T, 3}, B3)

        @test @inferred(B1 .+ sin.(B2 .- 3)) ≈ A1 .+ sin.(A2 .- 3)
        @test @inferred(B2' .* B3 .- Ref(0.5)) ≈ A2' .* A3 .- Ref(0.5)
        @test @inferred(B2' .* B3 .- max.(abs.(B1), real.(B3))) ≈
            A2' .* A3 .- max.(abs.(A1), real.(A3))

        @test (B1 .+ sin.(B2 .- 3)) isa StridedView
        @test (B2' .* B3 .- Ref(0.5)) isa StridedView
        @test (B2' .* B3 .- max.(abs.(B1), real.(B3))) isa StridedView
        @test (B2' .* A3 .- max.(abs.(B1), real.(B3))) isa Array
    end
end

@testset "broadcast with zero-length StridedView" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A1 = StridedView(zeros(T, (2, 0)))
        A2 = StridedView(zeros(T, (2, 0)))
        @test (A1 .+ A2) == StridedView(zeros(T, (2, 0)))
    end
end

@testset "mapreduce with StridedView" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        R1 = rand(T, (10, 10, 10, 10, 10, 10))
        @test sum(R1; dims = (1, 3, 5)) ≈ sum(StridedView(R1); dims = (1, 3, 5))
        @test mapreduce(sin, +, R1; dims = (1, 3, 5)) ≈
            mapreduce(sin, +, StridedView(R1); dims = (1, 3, 5))
        R2 = rand(T, (10, 10, 10))
        R2c = copy(R2)
        @test Strided._mapreducedim!(
            sin, +, identity, (10, 10, 10, 10, 10, 10),
            (
                sreshape(StridedView(R2c), (10, 1, 1, 10, 10, 1)),
                StridedView(R1),
            )
        ) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6)) .+ reshape(R2, (10, 1, 1, 10, 10, 1))
        R2c = copy(R2)
        @test Strided._mapreducedim!(
            sin, +, x -> 0, (10, 10, 10, 10, 10, 10),
            (
                sreshape(StridedView(R2c), (10, 1, 1, 10, 10, 1)),
                StridedView(R1),
            )
        ) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6))
        R2c = copy(R2)
        β = rand(T)
        @test Strided._mapreducedim!(
            sin, +, x -> β * x, (10, 10, 10, 10, 10, 10),
            (
                sreshape(StridedView(R2c), (10, 1, 1, 10, 10, 1)),
                StridedView(R1),
            )
        ) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6)) .+
            β .* reshape(R2, (10, 1, 1, 10, 10, 1))
        R2c = copy(R2)
        @test Strided._mapreducedim!(
            sin, +, x -> β, (10, 10, 10, 10, 10, 10),
            (
                sreshape(StridedView(R2c), (10, 1, 1, 10, 10, 1)),
                StridedView(R1),
            )
        ) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6), init = β)
        R2c = copy(R2)
        @test Strided._mapreducedim!(
            sin, +, conj, (10, 10, 10, 10, 10, 10),
            (
                sreshape(StridedView(R2c), (10, 1, 1, 10, 10, 1)),
                StridedView(R1),
            )
        ) ≈
            mapreduce(sin, +, R1; dims = (2, 3, 6)) .+
            conj.(reshape(R2, (10, 1, 1, 10, 10, 1)))

        R3 = rand(T, (100, 100, 2))
        @test sum(R3; dims = (1, 2)) ≈ sum(StridedView(R3); dims = (1, 2))
    end
end

@testset "complete reductions with StridedView" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        R1 = rand(T, (10, 10, 10, 10, 10, 10))

        @test sum(R1) ≈ sum(StridedView(R1))
        @test maximum(abs, R1) ≈ maximum(abs, StridedView(R1))
        @test minimum(real, R1) ≈ minimum(real, StridedView(R1))
        @test sum(x -> real(x) < 0, R1) == sum(x -> real(x) < 0, StridedView(R1))

        R2 = PermutedDimsArray(R1, (randperm(6)...,))

        @test sum(R2) ≈ sum(StridedView(R2))
        @test maximum(abs, R2) ≈ maximum(abs, StridedView(R2))
        @test minimum(real, R2) ≈ minimum(real, StridedView(R2))
        @test sum(x -> real(x) < 0, R1) == sum(x -> real(x) < 0, StridedView(R2))

        R3 = rand(T, (5, 5, 5))
        @test prod(exp, StridedView(R3)) ≈ exp(sum(StridedView(R3)))
    end
end

@testset "@strided macro" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A1, A2, A3 = rand(T, (10,)), rand(T, (10, 10)), rand(T, (10, 10, 10))

        @test (@strided(@. A1 + sin(A2 - 3))) isa StridedView
        @test (@strided(A1 .+ sin.(A2 .- 3))) isa StridedView
        @test (@strided(A1 .+ sin.(A2 .- 3))) ≈ A1 .+ sin.(A2 .- 3)
        @test (@strided(A2' .* A3 .- Ref(0.5))) ≈ A2' .* A3 .- Ref(0.5)
        @test (@strided(A2' .* A3 .- max.(abs.(A1), real.(A3)))) ≈
            A2' .* A3 .- max.(abs.(A1), real.(A3))

        B2 = view(A2, :, 1:2:10)
        @test (@strided(A1 .+ sin.(view(A2, :, 1:2:10) .- 3))) ≈
            (@strided(A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(view(A2, :, 1:2:10) .- 3)

        B2 = view(A2', :, 1:6)
        B3 = view(A3, :, 1:6, 4)
        @test (@strided(view(A2', :, 1:6) .* view(A3, :, 1:6, 4) .- Ref(0.5))) ≈
            (@strided(B2 .* B3 .- Ref(0.5))) ≈
            view(A2', :, 1:6) .* view(A3, :, 1:6, 4) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = view(A3, 1:5, :, 2:2:10)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (
                @strided(
                    view(A2, :, 3)' .* view(A3, 1:5, :, 2:2:10) .-
                    max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))
                )
            ) ≈
            (@strided(B2' .* B3 .- max.(abs.(B1), real.(B3b)))) ≈
            view(A2, :, 3)' .* view(A3, 1:5, :, 2:2:10) .-
            max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))

        B2 = reshape(A2, (10, 2, 5))
        @test (@strided(A1 .+ sin.(reshape(A2, (10, 2, 5)) .- 3))) ≈
            (@strided(A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(reshape(A2, (10, 2, 5)) .- 3)

        B2 = reshape(A2, 1, 100)
        B3 = reshape(A3, 100, 1, 10)
        @test (@strided(reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5))) ≈
            (@strided(B2' .* B3 .- Ref(0.5))) ≈
            reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = reshape(view(A3, 1:5, :, :), 5, 10, 5, 2)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (
                @strided(
                    view(A2, :, 3)' .* reshape(view(A3, 1:5, :, :), 5, 10, 5, 2) .-
                    max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))
                )
            ) ≈
            (@strided(B2' .* B3 .- max.(abs.(B1), real.(B3b)))) ≈
            view(A2, :, 3)' .* reshape(view(A3, 1:5, :, :), 5, 10, 5, 2) .-
            max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))

        x = @strided begin
            p = :A => A1
            f = pair -> (pair.first, pair.second)
            f(p)
        end
        @test x[2] === parent(StridedView(A1))
    end
end

@testset "@unsafe_strided macro" begin
    @testset for T in (Float32, Float64, ComplexF32, ComplexF64)
        A1, A2, A3 = rand(T, (10,)), rand(T, (10, 10)), rand(T, (10, 10, 10))

        @test (@unsafe_strided(A1, A2, @. A1 + sin(A2 - 3))) isa StridedView
        @test (@unsafe_strided(A1, A2, A1 .+ sin.(A2 .- 3))) isa StridedView

        @test (@unsafe_strided(A1, A2, A1 .+ sin.(A2 .- 3))) ≈ A1 .+ sin.(A2 .- 3)
        @test (@unsafe_strided(A2, A3, A2' .* A3 .- Ref(0.5))) ≈ A2' .* A3 .- Ref(0.5)
        @test (@unsafe_strided(A1, A2, A3, A2' .* A3 .- max.(abs.(A1), real.(A3)))) ≈
            A2' .* A3 .- max.(abs.(A1), real.(A3))

        B2 = view(A2, :, 1:2:10)
        @test (@unsafe_strided(A1, A2, A1 .+ sin.(view(A2, :, 1:2:10) .- 3))) ≈
            (@unsafe_strided(A1, B2, A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(view(A2, :, 1:2:10) .- 3)

        B2 = view(A2', :, 1:6)
        B3 = view(A3, :, 1:6, 4)
        @test (
                @unsafe_strided(
                    A2, A3,
                    view(A2', :, 1:6) .* view(A3, :, 1:6, 4) .- Ref(0.5)
                )
            ) ≈
            (@unsafe_strided(B2, B3, B2 .* B3 .- Ref(0.5))) ≈
            view(A2', :, 1:6) .* view(A3, :, 1:6, 4) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = view(A3, 1:5, :, 2:2:10)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (
                @unsafe_strided(
                    A1, A2, A3,
                    view(A2, :, 3)' .* view(A3, 1:5, :, 2:2:10) .-
                    max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))
                )
            ) ≈
            (@unsafe_strided(B1, B2, B3, B2' .* B3 .- max.(abs.(B1), real.(B3b)))) ≈
            view(A2, :, 3)' .* view(A3, 1:5, :, 2:2:10) .-
            max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))

        B2 = reshape(A2, (10, 2, 5))
        @test (@unsafe_strided(A1, A2, A1 .+ sin.(reshape(A2, (10, 2, 5)) .- 3))) ≈
            (@unsafe_strided(A1, B2, A1 .+ sin.(B2 .- 3))) ≈
            A1 .+ sin.(reshape(A2, (10, 2, 5)) .- 3)

        B2 = reshape(A2, 1, 100)
        B3 = reshape(A3, 100, 1, 10)
        @test (
                @unsafe_strided(
                    A2, A3,
                    reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5)
                )
            ) ≈
            (@unsafe_strided(B2, B3, B2' .* B3 .- Ref(0.5))) ≈
            reshape(A2, 1, 100)' .* reshape(A3, 100, 1, 10) .- Ref(0.5)

        B2 = view(A2, :, 3)
        B3 = reshape(view(A3, 1:5, :, :), 5, 10, 5, 2)
        B1 = view(A1, 1:5)
        B3b = view(A3, 4:4, 4:4, 2:2:10)
        @test (
                @unsafe_strided(
                    A1, A2, A3,
                    view(A2, :, 3)' .*
                    reshape(view(A3, 1:5, :, :), 5, 10, 5, 2) .-
                    max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))
                )
            ) ≈
            (@unsafe_strided(B1, B2, B3, B2' .* B3 .- max.(abs.(B1), real.(B3b)))) ≈
            view(A2, :, 3)' .* reshape(view(A3, 1:5, :, :), 5, 10, 5, 2) .-
            max.(abs.(view(A1, 1:5)), real.(view(A3, 4:4, 4:4, 2:2:10)))
    end
end

@testset "multiplication with StridedView: Complex{Int}" begin
    d = 103
    A1 = map(complex, rand(-100:100, (d, d)), rand(-100:100, (d, d)))
    A2 = map(complex, rand(-100:100, (d, d)), rand(-100:100, (d, d)))
    A3 = map(complex, rand(-100:100, (d, d)), rand(-100:100, (d, d)))
    A4 = map(complex, rand(-100:100, (d, d)), rand(-100:100, (d, d)))
    A1c = copy(A1)
    A2c = copy(A2)
    A3c = copy(A3)
    A4c = copy(A4)
    B1 = StridedView(A1c)
    B2 = StridedView(A2c)
    B3 = StridedView(A3c)
    B4 = StridedView(A4c)

    for op1 in (identity, conj, transpose, adjoint)
        @test op1(A1) == op1(B1)
        for op2 in (identity, conj, transpose, adjoint)
            @test op1(A1) * op2(A2) ≈ op1(B1) * op2(B2)
            for op3 in (identity, conj, transpose, adjoint)
                α = 2 + im
                β = 3 - im
                copy!(B3, B4)
                mul!(op3(B3), op1(B1), op2(B2), α, β)
                @test B3 ≈ op3(β) * A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                copy!(B3, B4)
                mul!(op3(B3), op1(B1), op2(B2), α, 0)
                @test B3 ≈ op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                copy!(B3, B4)
                mul!(op3(B3), op1(B1), op2(B2))
                @test B3 ≈ op3(op1(A1) * op2(A2)) # op3 is its own inverse
            end
        end
    end

    A = map(complex, rand(-100:100, (2, 0)), rand(-100:100, (2, 0)))
    B = map(complex, rand(-100:100, (0, 2)), rand(-100:100, (0, 2)))
    C = map(complex, rand(-100:100, (2, 2)), rand(-100:100, (2, 2)))
    α = complex(rand(-100:100), rand(-100:100))
    β = one(eltype(C))
    A1 = StridedView(copy(A))
    B1 = StridedView(copy(B))
    C1 = StridedView(copy(C))
    @test mul!(C, A, B, α, β) ≈ mul!(C1, A1, B1, α, β)
end

@testset "multiplication with StridedView: Rational{Int}" begin
    d = 103
    A1 = map(//, rand(-10:10, (d, d)), rand(1:10, (d, d)))
    A2 = map(//, rand(-10:10, (d, d)), rand(1:10, (d, d)))
    A3 = map(//, rand(-10:10, (d, d)), rand(1:10, (d, d)))
    A4 = map(//, rand(-10:10, (d, d)), rand(1:10, (d, d)))
    A1c = copy(A1)
    A2c = copy(A2)
    A3c = copy(A3)
    A4c = copy(A4)
    B1 = StridedView(A1c)
    B2 = StridedView(A2c)
    B3 = StridedView(A3c)
    B4 = StridedView(A4c)

    for op1 in (identity, conj, transpose, adjoint)
        @test op1(A1) == op1(B1)
        for op2 in (identity, conj, transpose, adjoint)
            @test op1(A1) * op2(A2) ≈ op1(B1) * op2(B2)
            for op3 in (identity, conj, transpose, adjoint)
                α = 1 // 2
                β = 3 // 2
                copy!(B3, B4)
                mul!(op3(B3), op1(B1), op2(B2), α, β)
                @test B3 ≈ op3(β) * A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                copy!(B3, B4)
                mul!(op3(B3), op1(B1), op2(B2), α, 1)
                @test B3 ≈ A4 + op3(α * op1(A1) * op2(A2)) # op3 is its own inverse
                copy!(B3, B4)
                mul!(op3(B3), op1(B1), op2(B2), 1, 1)
                @test B3 ≈ A4 + op3(op1(A1) * op2(A2)) # op3 is its own inverse
            end
        end
    end
end
