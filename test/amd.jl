for T in (Float32, Float64, Complex{Float32}, Complex{Float64})
    m1 = 32
    m2 = 16
    @testset "Copy with ROCStridedView: $T, $f1, $f2" for f2 in (identity, conj, adjoint, transpose), f1 in (identity, conj, transpose, adjoint)
        A1 = AMDGPU.randn(T, (m1, m2))
        A2 = similar(A1)
        A1c = copy(A1)
        A2c = copy(A2)
        B1 = f1(StridedView(A1c))
        B2 = f2(StridedView(A2c))
        axes(f1(A1)) == axes(f2(A2)) || continue
        @test collect(ROCMatrix(copy!(f2(A2), f1(A1)))) == Adapt.adapt(Vector{T}, copy!(B2, B1))
    end
end
