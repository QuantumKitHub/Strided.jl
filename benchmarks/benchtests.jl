using Revise
using LinearAlgebra
using BenchmarkTools
using Strided
using TensorOperations
using Tullio
using LoopVectorization

sizes = ceil.(Int, 2 .^ (2:1.5:20))

function benchmark_sum(sizes)
    times = zeros(length(sizes), 3)
    for (i, s) in enumerate(sizes)
        A = randn(Float64, s)
        times[i, 1] = @belapsed sum($A)
        Strided.disable_threads()
        times[i, 2] = @belapsed @strided sum($A)
        Strided.enable_threads()
        times[i, 3] = @belapsed @strided sum($A)
        println("step $i: size $s => times = $(times[i, :])")
    end
    return times
end

function benchmark_permute(sizes, p = (4, 3, 2, 1))
    times = zeros(length(sizes), 4)
    for (i, s) in enumerate(sizes)
        A = randn(Float64, s .* one.(p))
        B = similar(A)
        times[i, 1] = @belapsed copy!($B, $A)
        times[i, 2] = @belapsed permutedims!($B, $A, $p)
        Strided.disable_threads()
        times[i, 3] = @belapsed @strided permutedims!($B, $A, $p)
        Strided.enable_threads()
        times[i, 4] = @belapsed @strided permutedims!($B, $A, $p)
        println("step $i: size $s => times = $(times[i, :])")
    end
    return times
end
permute_times1 = benchmark_permute(sizes, (4, 3, 2, 1))
permute_times2 = benchmark_permute(sizes, (2, 3, 4, 1))
permute_times3 = benchmark_permute(sizes, (3, 4, 1, 2))

function benchmark_mul(sizesm, sizesk = sizesm, sizesn = sizesm)
    N = Threads.nthreads()
    @assert length(sizesm) == length(sizesk) == length(sizesn)
    times = zeros(length(sizesm), 4)
    @inbounds for i in 1:length(sizesm)
        m = sizesm[i]
        k = sizesk[i]
        n = sizesn[i]
        A = randn(Float64, (m, k))
        B = randn(Float64, (k, n))
        C = randn(Float64, (m, n))

        BLAS.set_num_threads(1) # base case: single-threaded blas
        times[i, 1] = @belapsed mul!($C, $A, $B)
        BLAS.set_num_threads(N) # multithreaded blas
        times[i, 2] = @belapsed mul!($C, $A, $B)
        Strided.disable_threaded_mul() # same, except for small overhead from strided
        times[i, 3] = @belapsed @strided mul!($C, $A, $B)
        BLAS.set_num_threads(1) # single-threaded blas with strided multithreading
        Strided.enable_threaded_mul()
        times[i, 4] = @belapsed @strided mul!($C, $A, $B)
        println("step $i: sizes $((m, k, n)) => times = $(times[i, :])")
    end
    return times
end

function tensorcontraction!(wEnv, hamAB, hamBA, rhoBA, rhoAB, w, v, u)
    @tensor wEnv[-1, -2, -3] = hamAB[7, 8, -1, 9] * rhoBA[4, 3, -3, 2] * conj(w[7, 5, 4]) *
        u[9, 10, -2, 11] * conj(u[8, 10, 5, 6]) * v[1, 11, 2] *
        conj(v[1, 6, 3]) +
        hamBA[1, 2, 3, 4] * rhoBA[10, 7, -3, 6] *
        conj(w[-1, 11, 10]) * u[3, 4, -2, 8] * conj(u[1, 2, 11, 9]) *
        v[5, 8, 6] * conj(v[5, 9, 7]) +
        hamAB[5, 7, 3, 1] * rhoBA[10, 9, -3, 8] *
        conj(w[-1, 11, 10]) * u[4, 3, -2, 2] * conj(u[4, 5, 11, 6]) *
        v[1, 2, 8] * conj(v[7, 6, 9]) +
        hamBA[3, 7, 2, -1] * rhoAB[5, 6, 4, -3] * v[2, 1, 4] *
        conj(v[3, 1, 5]) * conj(w[7, -2, 6])
    return wEnv
end

function benchmark_tensorcontraction(sizes)
    N = Threads.nthreads()
    times = zeros(length(sizes), 5)
    @inbounds for (i, s) in enumerate(sizes)
        hamAB = randn(Float64, (s, s, s, s))
        hamBA = randn(Float64, (s, s, s, s))
        rhoAB = randn(Float64, (s, s, s, s))
        rhoBA = randn(Float64, (s, s, s, s))
        v = randn(Float64, (s, s, s))
        w = randn(Float64, (s, s, s))
        u = randn(Float64, (s, s, s, s))
        wEnv = randn(Float64, (s, s, s))

        tensorcontraction!(wEnv, hamAB, hamBA, rhoBA, rhoAB, w, v, u)

        BLAS.set_num_threads(1)
        Strided.disable_threads()
        Strided.disable_threaded_mul()
        times[i, 1] = @belapsed tensorcontraction!(
            $wEnv, $hamAB, $hamBA, $rhoBA, $rhoAB,
            $w, $v, $u
        )

        BLAS.set_num_threads(1)
        Strided.enable_threads()
        Strided.disable_threaded_mul()
        times[i, 2] = @belapsed tensorcontraction!(
            $wEnv, $hamAB, $hamBA, $rhoBA, $rhoAB,
            $w, $v, $u
        )

        BLAS.set_num_threads(N)
        Strided.disable_threads()
        Strided.disable_threaded_mul()
        times[i, 3] = @belapsed tensorcontraction!(
            $wEnv, $hamAB, $hamBA, $rhoBA, $rhoAB,
            $w, $v, $u
        )

        BLAS.set_num_threads(N)
        Strided.enable_threads()
        Strided.disable_threaded_mul()
        times[i, 4] = @belapsed tensorcontraction!(
            $wEnv, $hamAB, $hamBA, $rhoBA, $rhoAB,
            $w, $v, $u
        )

        BLAS.set_num_threads(1)
        Strided.enable_threads()
        Strided.enable_threaded_mul()
        times[i, 5] = @belapsed tensorcontraction!(
            $wEnv, $hamAB, $hamBA, $rhoBA, $rhoAB,
            $w, $v, $u
        )

        println("step $i: size $s => times = $(times[i, :])")
    end
    return times
end
