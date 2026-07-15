using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    Ts = [Float32, Float64, ComplexF32, ComplexF64]
    blasops = [identity, conj, transpose, adjoint]
    @compile_workload begin
        for T in Ts
            A = StridedView(zeros(T, 2, 2))
            B = StridedView(zeros(T, 2, 2))
            C = StridedView(zeros(T, 2, 2))
            # op variety on the operands, both scalar-argument forms
            for opA in blasops, opB in blasops
                mul!(C, opA(A), opB(B))
                mul!(C, opA(A), opB(B), one(T), zero(T))
            end
            # op variety on the destination (mul! normalization branches)
            for opC in (transpose, adjoint)
                mul!(opC(C), A, B)
                mul!(opC(C), A, B, one(T), zero(T))
            end
        end
    end
end
