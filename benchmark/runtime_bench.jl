# Runtime benchmark.
#
# Measures steady-state (compiled) performance of the mapreduce machinery, so we
# can guard against runtime regressions — permutations especially. Runs each
# case single-threaded and (if available) multi-threaded.
#
#   julia --project=benchmark -t auto benchmark/runtime_bench.jl [label]
#
# Writes results to benchmark/results/runtime_<label>.tsv

include(joinpath(@__DIR__, "setup_env.jl"))
include(joinpath(@__DIR__, "cases.jl"))
using BenchmarkTools

const LABEL = length(ARGS) >= 1 ? ARGS[1] : "baseline"

# Large enough that the kernel, not call overhead, dominates.
const BIG_TOTAL = 1 << 22   # ~4M elements

function bench_one(c::Case)
    sz = sizetuple(c.N, BIG_TOTAL)
    run = make_runner(c, sz)
    run()                                    # warm up / compile
    return @belapsed $run() samples = 30 evals = 1
end

function main()
    cases = all_cases(;
        Ns = 2:6,
        Ts = (Float64, ComplexF64),
        kinds = (permute, add, reduce_inner, reduce_outer, reduce_full),
    )

    nthreads_available = Base.Threads.nthreads()
    thread_settings = nthreads_available > 1 ? (1, nthreads_available) : (1,)

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "runtime_$(LABEL).tsv")
    io = open(out, "w")
    println(io, "case\tnthreads\ttime_us")

    for nt in thread_settings
        Strided.set_num_threads(nt)
        println("== runtime [$LABEL] nthreads=$nt ==")
        for c in cases
            t = bench_one(c)
            us = t * 1e6
            println(io, "$(name(c))\t$nt\t$(round(us; digits = 3))")
            println(rpad(name(c), 32), "  nt=$nt  ", round(us; digits = 3), " us")
        end
    end
    close(io)
    println("\nwrote $out")
end

main()
