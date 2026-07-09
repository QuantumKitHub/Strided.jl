# Benchmark suite for Strided.jl, in the standard BenchmarkTools.jl format
# expected by AirSpeedVelocity.jl / PkgBenchmark.jl: this file defines a
# `SUITE::BenchmarkGroup` which the harness loads and runs for each revision.
# This file never runs the suite itself.
#
# The suite is configured through command-line arguments (see --help). When
# included by a harness (empty ARGS), the default configuration is used: the
# full suite, which takes on the order of an hour per revision. Use the
# harness' own filtering (e.g. `benchpkg --filter`) to restrict it. For a
# manual run, arguments can be passed after `--`:
#
#   julia --project=benchmark -t 8 \
#       -e 'include("benchmark/benchmarks.jl"); display(run(SUITE; verbose=true))' \
#       -- --groups=small --eltypes=Float64
#
# The suite focuses on `permutedims!` of strided arrays. The cases are grouped
# and defined in a TOML file, by default the cases.toml next to this script
# (see its comments for the format and the provenance of the default groups:
# hptt, small, unaligned); a different file can be passed with --cases.
#
# Structure of the suite:
#   SUITE["permutedims!"][group]["T=$T"]["nthreads=$nt"]["p=$(p)_sz=$(sz)"]

using ArgParse
using BenchmarkTools
using Strided
using Strided: StridedView
using TOML

function parse_config(args)
    s = ArgParseSettings(;
        prog = "benchmark/benchmarks.jl",
        description = "Benchmark suite for permutedims! on strided arrays; " *
            "defines `SUITE` without running it.",
    )
    #! format: off
    @add_arg_table! s begin
        "--eltypes", "-T"
            help = "comma-separated element types to benchmark"
            default = "Float64,ComplexF64"
        "--cases", "-c"
            help = "TOML file defining the benchmark cases"
            default = joinpath(@__DIR__, "cases.toml")
        "--groups", "-g"
            help = "comma-separated case groups to benchmark; " *
                   "defaults to all groups in the cases file"
            default = ""
        "--nthreads", "-n"
            help = "comma-separated Strided thread counts; values above the " *
                   "number of Julia threads (set with julia -t) are clamped"
            default = "1,$(Threads.nthreads())"
        "--filter", "-f"
            help = "only include benchmarks whose full name contains this substring"
            default = ""
    end
    #! format: on
    return parse_args(args, s)
end

const CONFIG = parse_config(ARGS)

const ELTYPES = map(split(CONFIG["eltypes"], ',')) do s
    T = getfield(Base, Symbol(strip(s)))
    T isa Type || error("--eltypes: `$s` is not a type")
    return T
end
const NTHREADS = sort!(
    unique(
        clamp.(
            parse.(Int, strip.(split(CONFIG["nthreads"], ','))), 1, Threads.nthreads()
        )
    )
)

# Arrays shorter than this are never threaded by Strided's kernel, so threaded
# variants of such cases would just duplicate the single-threaded numbers.
const MINTHREADLENGTH = isdefined(Strided, :MINTHREADLENGTH) ? Strided.MINTHREADLENGTH : 1 << 15

# The benchmark cases, as a Dict mapping group name => group spec (sampling
# parameters and case list).
const CASEGROUPS = TOML.parsefile(CONFIG["cases"])
const GROUPS = if isempty(strip(CONFIG["groups"]))
    sort!(collect(keys(CASEGROUPS)))
else
    strip.(split(CONFIG["groups"], ','))
end

function parsecases(spec, path)
    return map(spec["cases"]) do c
        p = (Int.(c["p"])...,)
        sz = (Int.(c["size"])...,)
        (length(p) == length(sz) && isperm(p)) ||
            error("$path: `p = $(c["p"])` is not a permutation matching `size = $(c["size"])`")
        return (p, sz)
    end
end

casename(p, sz) = "p=" * join(p) * "_sz=" * join(sz, 'x')

# Arrays are allocated (and touched) in the per-sample setup rather than at
# suite-construction time: with the full grid of cases the arrays would
# otherwise all be alive at once, at up to ~0.9 GB each. `evals` is fixed so
# that tuning never re-runs the expensive setup, and is > 1 only for cases too
# short to time reliably in a single evaluation.
function addcases!(group::BenchmarkGroup, cases, T, nt; samples, seconds)
    for (p, sz) in cases
        nt > 1 && prod(sz) < MINTHREADLENGTH && continue
        dsz = getindex.(Ref(sz), p)
        evals = max(1, (1 << 15) ÷ prod(sz))
        group[casename(p, sz)] = @benchmarkable(
            permutedims!(StridedView(B), StridedView(A), $p),
            setup = (
                Strided.set_num_threads($nt);
                A = rand($T, $sz);
                B = fill!(Array{$T}(undef, $dsz), zero($T))
            ),
            evals = evals,
            samples = samples,
            seconds = seconds,
        )
    end
    return group
end

function filtered(group::BenchmarkGroup, pattern::AbstractString, prefix::String = "")
    out = BenchmarkGroup(group.tags)
    for (key, value) in group
        name = prefix * "/" * key
        if value isa BenchmarkGroup
            sub = filtered(value, pattern, name)
            isempty(sub.data) || (out[key] = sub)
        elseif occursin(pattern, name)
            out[key] = value
        end
    end
    return out
end

const SUITE = BenchmarkGroup()
let permute = addgroup!(SUITE, "permutedims!")
    for name in GROUPS
        haskey(CASEGROUPS, name) ||
            error("--groups: group `$name` not found in $(CONFIG["cases"])")
        spec = CASEGROUPS[name]
        cases = parsecases(spec, CONFIG["cases"])
        samples, seconds = spec["samples"], spec["seconds"]
        g = addgroup!(permute, name)
        for T in ELTYPES
            tg = addgroup!(g, "T=$T")
            for nt in NTHREADS
                addcases!(addgroup!(tg, "nthreads=$nt"), cases, T, nt; samples, seconds)
            end
        end
    end
    if !isempty(CONFIG["filter"])
        SUITE["permutedims!"] = filtered(permute, CONFIG["filter"])
    end
end
