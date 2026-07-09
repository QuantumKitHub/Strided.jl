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
# The suite focuses on `permutedims!` of strided arrays, with three groups of
# cases:
#   * "hptt": the reference benchmark of HPTT / the TTC paper
#     (https://github.com/springer13/hptt/blob/master/benchmark/benchmark.sh),
#     reproduced verbatim at the original (large, bandwidth-bound) sizes, with
#     the 0-based permutations translated to 1-based.
#   * "small": small to medium sizes, including the permutation cases of
#     strided-rs (https://github.com/tensor4all/strided-rs), to track per-call
#     overhead, dispatch cost, and the non-threaded fast paths.
#   * "unaligned": non-SIMD-aligned sizes (powers of two ± 1, primes, mixed
#     odd shapes) that exercise vector-remainder loops and blocking edges.
#
# Structure of the suite:
#   SUITE["permutedims!"][group]["T=$T"]["nthreads=$nt"]["p=$(p)_sz=$(sz)"]

using ArgParse
using BenchmarkTools
using Strided
using Strided: StridedView

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
        "--groups", "-g"
            help = "comma-separated case groups to benchmark (hptt, small, unaligned)"
            default = "hptt,small,unaligned"
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
const GROUPS = strip.(split(CONFIG["groups"], ','))
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

# The HPTT reference benchmark: (permutation, input size). Each permutation
# comes in three size variants: balanced dims, large first dim, large last dim.
const HPTT_CASES = [
    # 2D
    ((2, 1), (7264, 7264)),
    ((2, 1), (43408, 1216)),
    ((2, 1), (1216, 43408)),
    # 3D
    ((1, 3, 2), (368, 384, 384)),
    ((1, 3, 2), (2144, 64, 384)),
    ((1, 3, 2), (368, 64, 2307)),
    ((2, 1, 3), (384, 384, 355)),
    ((2, 1, 3), (2320, 384, 59)),
    ((2, 1, 3), (384, 2320, 59)),
    ((3, 2, 1), (384, 355, 384)),
    ((3, 2, 1), (2320, 59, 384)),
    ((3, 2, 1), (384, 59, 2320)),
    # 4D
    ((1, 4, 3, 2), (80, 96, 75, 96)),
    ((1, 4, 3, 2), (464, 16, 75, 96)),
    ((1, 4, 3, 2), (80, 16, 75, 582)),
    ((3, 2, 4, 1), (96, 75, 96, 75)),
    ((3, 2, 4, 1), (608, 12, 96, 75)),
    ((3, 2, 4, 1), (96, 12, 608, 75)),
    ((3, 1, 4, 2), (96, 75, 96, 75)),
    ((3, 1, 4, 2), (608, 12, 96, 75)),
    ((3, 1, 4, 2), (96, 12, 608, 75)),
    ((2, 1, 4, 3), (96, 96, 75, 75)),
    ((2, 1, 4, 3), (608, 96, 12, 75)),
    ((2, 1, 4, 3), (96, 608, 12, 75)),
    ((4, 3, 2, 1), (96, 75, 75, 96)),
    ((4, 3, 2, 1), (608, 12, 75, 96)),
    ((4, 3, 2, 1), (96, 12, 75, 608)),
    # 5D
    ((1, 5, 3, 2, 4), (32, 48, 28, 28, 48)),
    ((1, 5, 3, 2, 4), (176, 8, 28, 28, 48)),
    ((1, 5, 3, 2, 4), (32, 8, 28, 28, 298)),
    ((4, 3, 2, 5, 1), (48, 28, 28, 48, 28)),
    ((4, 3, 2, 5, 1), (352, 4, 28, 48, 28)),
    ((4, 3, 2, 5, 1), (48, 4, 28, 352, 28)),
    ((3, 1, 5, 2, 4), (48, 28, 48, 28, 28)),
    ((3, 1, 5, 2, 4), (352, 4, 48, 28, 28)),
    ((3, 1, 5, 2, 4), (48, 4, 352, 28, 28)),
    ((2, 4, 1, 5, 3), (48, 48, 28, 28, 28)),
    ((2, 4, 1, 5, 3), (352, 48, 4, 28, 28)),
    ((2, 4, 1, 5, 3), (48, 352, 4, 28, 28)),
    ((5, 4, 3, 2, 1), (48, 28, 28, 28, 48)),
    ((5, 4, 3, 2, 1), (352, 4, 28, 28, 48)),
    ((5, 4, 3, 2, 1), (48, 4, 28, 28, 352)),
    # 6D
    ((1, 4, 3, 6, 5, 2), (16, 32, 15, 32, 15, 15)),
    ((1, 4, 3, 6, 5, 2), (48, 10, 15, 32, 15, 15)),
    ((1, 4, 3, 6, 5, 2), (16, 10, 15, 103, 15, 15)),
    ((4, 3, 1, 6, 2, 5), (32, 15, 15, 32, 15, 15)),
    ((4, 3, 1, 6, 2, 5), (112, 5, 15, 32, 15, 15)),
    ((4, 3, 1, 6, 2, 5), (32, 5, 15, 112, 15, 15)),
    ((3, 1, 5, 2, 6, 4), (32, 15, 32, 15, 15, 15)),
    ((3, 1, 5, 2, 6, 4), (112, 5, 32, 15, 15, 15)),
    ((3, 1, 5, 2, 6, 4), (32, 5, 112, 15, 15, 15)),
    ((4, 3, 6, 2, 1, 5), (32, 15, 15, 32, 15, 15)),
    ((4, 3, 6, 2, 1, 5), (112, 5, 15, 32, 15, 15)),
    ((4, 3, 6, 2, 1, 5), (32, 5, 15, 112, 15, 15)),
    ((6, 5, 4, 3, 2, 1), (32, 15, 15, 15, 15, 32)),
    ((6, 5, 4, 3, 2, 1), (112, 5, 15, 15, 15, 32)),
    ((6, 5, 4, 3, 2, 1), (32, 5, 15, 15, 15, 112)),
]

# Small and medium sizes: per-call overhead and non-threaded fast paths, plus
# the permutation cases of strided-rs (1000², 4000², 32⁴ reversed).
const SMALL_CASES = [
    ((2, 1), (16, 16)),
    ((2, 1), (64, 64)),
    ((2, 1), (256, 256)),
    ((2, 1), (1000, 1000)),
    ((2, 1), (4000, 4000)),
    ((3, 2, 1), (16, 16, 16)),
    ((3, 2, 1), (64, 64, 64)),
    ((2, 3, 1), (16, 16, 16)),
    ((2, 3, 1), (64, 64, 64)),
    ((4, 3, 2, 1), (8, 8, 8, 8)),
    ((4, 3, 2, 1), (16, 16, 16, 16)),
    ((4, 3, 2, 1), (32, 32, 32, 32)),
    ((2, 1, 4, 3), (8, 8, 8, 8)),
    ((2, 1, 4, 3), (16, 16, 16, 16)),
    ((2, 1, 4, 3), (32, 32, 32, 32)),
    ((3, 4, 1, 2), (8, 8, 8, 8)),
    ((3, 4, 1, 2), (16, 16, 16, 16)),
    ((3, 4, 1, 2), (32, 32, 32, 32)),
    ((2, 3, 4, 1), (8, 8, 8, 8)),
    ((2, 3, 4, 1), (16, 16, 16, 16)),
    ((2, 3, 4, 1), (32, 32, 32, 32)),
]

# Sizes that are not multiples of the SIMD vector width: powers of two ± 1,
# primes, and mixed odd/rectangular shapes. These defeat aligned vectorization
# and exercise the vector-remainder loops and blocking edge cases that the
# power-of-two sizes of the other groups never hit.
const UNALIGNED_CASES = [
    ((2, 1), (17, 17)),
    ((2, 1), (63, 63)),
    ((2, 1), (129, 129)),
    ((2, 1), (251, 251)),
    ((2, 1), (999, 1001)),
    ((2, 1), (1021, 1021)),
    ((3, 2, 1), (15, 15, 15)),
    ((3, 2, 1), (17, 17, 17)),
    ((3, 2, 1), (63, 63, 63)),
    ((3, 2, 1), (65, 65, 65)),
    ((3, 2, 1), (31, 32, 33)),
    ((3, 2, 1), (37, 61, 89)),
    ((2, 3, 1), (15, 15, 15)),
    ((2, 3, 1), (17, 17, 17)),
    ((2, 3, 1), (63, 63, 63)),
    ((2, 3, 1), (65, 65, 65)),
    ((2, 3, 1), (31, 32, 33)),
    ((2, 3, 1), (37, 61, 89)),
    ((4, 3, 2, 1), (7, 7, 7, 7)),
    ((4, 3, 2, 1), (9, 9, 9, 9)),
    ((4, 3, 2, 1), (15, 15, 15, 15)),
    ((4, 3, 2, 1), (17, 17, 17, 17)),
    ((4, 3, 2, 1), (31, 31, 31, 31)),
    ((4, 3, 2, 1), (33, 33, 33, 33)),
    ((2, 1, 4, 3), (7, 7, 7, 7)),
    ((2, 1, 4, 3), (9, 9, 9, 9)),
    ((2, 1, 4, 3), (15, 15, 15, 15)),
    ((2, 1, 4, 3), (17, 17, 17, 17)),
    ((2, 1, 4, 3), (31, 31, 31, 31)),
    ((2, 1, 4, 3), (33, 33, 33, 33)),
    ((3, 4, 1, 2), (7, 7, 7, 7)),
    ((3, 4, 1, 2), (9, 9, 9, 9)),
    ((3, 4, 1, 2), (15, 15, 15, 15)),
    ((3, 4, 1, 2), (17, 17, 17, 17)),
    ((3, 4, 1, 2), (31, 31, 31, 31)),
    ((3, 4, 1, 2), (33, 33, 33, 33)),
    ((2, 3, 4, 1), (7, 7, 7, 7)),
    ((2, 3, 4, 1), (9, 9, 9, 9)),
    ((2, 3, 4, 1), (15, 15, 15, 15)),
    ((2, 3, 4, 1), (17, 17, 17, 17)),
    ((2, 3, 4, 1), (31, 31, 31, 31)),
    ((2, 3, 4, 1), (33, 33, 33, 33)),
]

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
        cases, samples, seconds = if name == "hptt"
            HPTT_CASES, 10, 10
        elseif name == "small"
            SMALL_CASES, 1000, 5
        elseif name == "unaligned"
            UNALIGNED_CASES, 1000, 5
        else
            error("--groups: unknown group `$name`")
        end
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
