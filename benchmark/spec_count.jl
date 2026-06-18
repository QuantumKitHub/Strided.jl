# Count method specializations of the bookkeeping functions after a
# multi-op / multi-eltype / multi-ndims workload. This is the headline
# precompile-effectiveness metric: fewer specializations => precompile once
# per N and reuse.
#
#   julia --project=benchmark benchmark/spec_count.jl [label]
include(joinpath(@__DIR__, "setup_env.jl"))
include(joinpath(@__DIR__, "cases.jl"))
using Strided

const LABEL = length(ARGS) >= 1 ? ARGS[1] : "baseline"

function nspecs(f)
    n = 0
    for m in methods(f)
        for s in Base.specializations(m)
            s === nothing && continue
            n += 1
        end
    end
    return n
end

# distinct map fns and reduce ops (distinct types)
const MAPFNS = Function[]
const REDFNS = Function[]
for i in 1:6
    push!(MAPFNS, @eval ($(Symbol(:mf_, i)))(x) = x * $i - $i)
    push!(REDFNS, @eval ($(Symbol(:rf_, i)))(x, y) = x + y * $(i % 3 + 1))
end
sz(N) = ntuple(_ -> 3, N)

function workload()
    Strided.set_num_threads(1)
    for N in 2:7
        for T in (Float64, ComplexF64, Float32, ComplexF32)
            for k in 1:6
                A = StridedView(rand(T, sz(N)))
                B = StridedView(zeros(T, sz(N)))
                map!(MAPFNS[k], B, A)
                r = StridedView(zeros(T, ntuple(i -> i == 1 ? 1 : 3, N)))
                Base.mapreducedim!(identity, REDFNS[k], r, A)
            end
        end
    end
end

function main()
    workload()
    fns = Dict(
        "_mapreduce_fuse!" => Strided._mapreduce_fuse!,
        "_mapreduce_order!" => Strided._mapreduce_order!,
        "_mapreduce_block!" => Strided._mapreduce_block!,
        "_computeblocks" => Strided._computeblocks,
        "_mapreduce_kernel!" => Strided._mapreduce_kernel!,
        "_mapreduce_threaded!" => Strided._mapreduce_threaded!,
        "indexorder" => Strided.indexorder,
        "totalmemoryregion" => Strided.totalmemoryregion,
    )
    mkpath(joinpath(@__DIR__, "results"))
    out = joinpath(@__DIR__, "results", "specs_$(LABEL).tsv")
    open(out, "w") do io
        println(io, "function\tnspecs")
        for nm in sort(collect(keys(fns)))
            n = nspecs(fns[nm])
            println(io, "$nm\t$n")
            println(rpad(nm, 24), "  ", n)
        end
    end
    println("wrote $out")
end
main()
