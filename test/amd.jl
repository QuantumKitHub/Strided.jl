for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
    @testset "Copy with ROCStridedView: $T, $f1, $f2" for f2 in (identity, conj, adjoint, transpose), f1 in (identity, conj, transpose, adjoint)
        for m1 in (0, 16, 32), m2 in (0, 16, 32)
            if iszero(m1 * m2)
                A1 = AMDGPU.ROCMatrix{T}(undef, (m1, m2))
            else
                A1 = ROCMatrix(randn(T, (m1, m2)))
            end
            A2 = similar(A1)
            zA1 = ROCMatrix(f1(zeros(T, (m1, m2))))
            zA2 = ROCMatrix(f2(zeros(T, (m1, m2))))
            A1c = copy(A1)
            A2c = copy(A2)
            B1 = f1(StridedView(A1c))
            B2 = f2(StridedView(A2c))
            axes(f1(A1)) == axes(f2(A2)) || continue
            @test collect(ROCMatrix(copy!(f2(A2), f1(A1)))) == AMDGPU.Adapt.adapt(Vector{T}, copy!(B2, B1))
            @test copy!(zA1, f1(A1)) == copy!(zA2, B1)
            x = rand(T)
            @test f1(StridedView(AMDGPU.Adapt.adapt(Vector{T}, fill!(A1c, x)))) == AMDGPU.Adapt.adapt(Vector{T}, fill!(B1, x))
        end
    end
end
