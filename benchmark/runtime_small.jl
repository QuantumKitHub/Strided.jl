# Small-array runtime benchmark: fixed bookkeeping overhead dominates here, so
# this is the sensitive guard against de-specialization regressions.
#
#   julia --project=benchmark -t 1 benchmark/runtime_small.jl [label]
include(joinpath(@__DIR__, "setup_env.jl"))
include(joinpath(@__DIR__, "cases.jl"))
using BenchmarkTools

const LABEL = length(ARGS) >= 1 ? ARGS[1] : "baseline"

function bench_one(c::Case, sz)
    run = make_runner(c, sz)
    run()
    return @belapsed $run() samples = 200 evals = 5
end

function main()
    Strided.set_num_threads(1)
    sizes = Dict(2 => (4, 4), 3 => (4, 4, 4), 4 => (4, 4, 4, 4))
    cases = all_cases(;
        Ns = 2:4,
        Ts = (Float64, ComplexF64),
        kinds = (permute, add, reduce_inner, reduce_outer, reduce_full),
    )
    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "runtime_small_$(LABEL).tsv")
    io = open(out, "w")
    println(io, "case\ttime_ns")
    println("== runtime small [$LABEL] nt=1 ==")
    for c in cases
        t = bench_one(c, sizes[c.N])
        ns = t * 1e9
        println(io, "$(name(c))\t$(round(ns; digits = 2))")
        println(rpad(name(c), 32), "  ", round(ns; digits = 2), " ns")
    end
    close(io)
    println("\nwrote $out")
end
main()
