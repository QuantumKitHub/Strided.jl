# Compile / TTFX benchmark.
#
# Run in a FRESH Julia process. Strided's kernels are not part of any precompile
# workload, so the first call to each (N, T, op) specialization triggers
# inference + codegen. `Base.@timed` reports `.compile_time` per call, which we
# sum across all cases to get the total cold-compile cost — the headline number
# we want to drive down.
#
#   julia --project=benchmark benchmark/compile_bench.jl [label]
#
# Writes results to benchmark/results/compile_<label>.tsv

include(joinpath(@__DIR__, "setup_env.jl"))
include(joinpath(@__DIR__, "cases.jl"))

const LABEL = length(ARGS) >= 1 ? ARGS[1] : "baseline"

# Small arrays: we want to isolate compile time, not run time.
const SMALL_TOTAL = 1 << 12   # 4096 elements

function main()
    cases = all_cases(;
        Ns = 2:7,
        Ts = (Float64, ComplexF64),
        kinds = (permute, add, reduce_inner, reduce_outer, reduce_full),
    )

    rows = Tuple{String,Float64,Float64}[]   # name, compile_time, total_time
    total_compile = 0.0
    for c in cases
        sz = sizetuple(c.N, SMALL_TOTAL)
        run = make_runner(c, sz)
        stats = Base.@timed run()            # first (cold) call
        push!(rows, (name(c), stats.compile_time, stats.time))
        total_compile += stats.compile_time
    end

    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "compile_$(LABEL).tsv")
    open(out, "w") do io
        println(io, "case\tcompile_s\ttotal_s")
        for (nm, ct, tt) in rows
            println(io, "$nm\t$(round(ct; digits = 5))\t$(round(tt; digits = 5))")
        end
        println(io, "TOTAL\t$(round(total_compile; digits = 5))\t")
    end

    println("== compile benchmark [$LABEL] ==")
    for (nm, ct, tt) in rows
        println(rpad(nm, 32), "  compile=", rpad(round(ct; digits = 4), 9), " total=", round(tt; digits = 4))
    end
    println("-"^56)
    println(rpad("TOTAL compile_time (s)", 32), "  ", round(total_compile; digits = 4))
    println("\nwrote $out")
end

main()
