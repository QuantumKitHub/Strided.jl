# Compile benchmark across MANY distinct op TYPES — simulating the real
# combinatorial explosion (TensorOperations generates many distinct map/reduce
# closures). Each `@eval`'d function is a distinct type, forcing a fresh
# specialization of the whole call chain.
#
#   julia --project=benchmark benchmark/manyops_compile.jl [label] [Kops] [Nmax]

include(joinpath(@__DIR__, "setup_env.jl"))
using Strided
using Strided: StridedView

const LABEL = length(ARGS) >= 1 ? ARGS[1] : "baseline"
const KOPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
const NMAXD = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5

# K distinct unary map functions (distinct types) and K distinct binary reduce ops.
const MAPFNS = Function[]
const REDFNS = Function[]
for i in 1:KOPS
    f = @eval ($(Symbol(:mapf_, i)))(x) = x * $i - $i
    g = @eval ($(Symbol(:redf_, i)))(x, y) = x + y * $(i % 3 + 1)
    push!(MAPFNS, f)
    push!(REDFNS, g)
end

sz(N) = ntuple(_ -> 3, N)

function main()
    total = 0.0
    Strided.set_num_threads(1)
    for N in 2:NMAXD
        for k in 1:KOPS
            A = StridedView(rand(Float64, sz(N)))
            B = StridedView(zeros(Float64, sz(N)))
            f = MAPFNS[k]
            total += Base.@timed(map!(f, B, A)).compile_time
            r = StridedView(zeros(Float64, ntuple(i -> i == 1 ? 1 : 3, N)))
            g = REDFNS[k]
            total += Base.@timed(Base.mapreducedim!(identity, g, r, A)).compile_time
        end
    end
    mkpath(joinpath(@__DIR__, "results"))
    open(joinpath(@__DIR__, "results", "manyops_$(LABEL).txt"), "w") do io
        println(io, "label=$LABEL Kops=$KOPS Nmax=$NMAXD total_compile_s=$(round(total; digits = 4))")
    end
    println("[$LABEL] Kops=$KOPS Nmax=$NMAXD  TOTAL compile_time = $(round(total; digits = 4)) s")
end

main()
