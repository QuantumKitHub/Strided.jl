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
#       -- --groups=balanced --eltypes=Float64
#
# The suite focuses on `permutedims!` of strided arrays. Each case fixes a
# *shape* (dimension ratios) and a *permutation*, and is swept over a list of
# total lengths; the cases are defined in a TOML file, by default the cases.toml
# next to this script (see its comments for the format and the groups). A
# different file can be passed with --cases.
#
# Alongside `permutedims!`, the suite auto-generates a "copy" group: a plain
# `copyto!` on `Vector`s of each swept length. Since a permutation and a copy
# both read and write the same number of elements, dividing a permutation's
# time by the copy time of the same length gives a machine-independent
# efficiency (>= 1, approaching 1 when bandwidth-bound). The copy baseline is
# single-threaded, so multi-threaded permutation ratios are relative to a
# single-threaded memory copy.
#
# Structure of the suite:
#   SUITE["permutedims!"][group]["T=$T"]["nthreads=$nt"][case]
#   SUITE["copy"]["T=$T"]["L=$L"]

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
        ),
    ),
)

# Arrays shorter than this are never threaded by Strided's kernel, so threaded
# variants of such cases would just duplicate the single-threaded numbers.
const MINTHREADLENGTH = isdefined(Strided, :MINTHREADLENGTH) ? Strided.MINTHREADLENGTH : 1 << 15

const CASEGROUPS = TOML.parsefile(CONFIG["cases"])
const GROUPS = if isempty(strip(CONFIG["groups"]))
    sort!(collect(keys(CASEGROUPS)))
else
    strip.(split(CONFIG["groups"], ','))
end

# Column-major strides of a dense array of size `sz`.
colstrides(sz::NTuple{N, Int}) where {N} = ntuple(i -> prod(ntuple(j -> sz[j], i - 1)), N)

# Dimensions realizing (approximately) `total` elements at the given shape ratio.
function shapedims(shape::NTuple{N, <:Real}, total::Int) where {N}
    c = (total / prod(shape))^(1 / N)
    return ntuple(i -> max(1, round(Int, shape[i] * c)), N)
end

# A StridedView of logical size `sz` whose axes are spaced by `mult` inside a
# freshly allocated (page-faulted) backing buffer. `mult` all ones gives a dense
# view (the unit-stride fast path); a leading value > 1 gives a non-unit
# innermost stride, a later value > 1 gives gaps between higher dimensions.
function make_view(::Type{T}, sz::NTuple{N, Int}, mult::NTuple{N, Int}) where {T, N}
    all(isone, mult) && return StridedView(rand(T, sz))
    bufsz = ntuple(i -> sz[i] * mult[i], N)
    strides = ntuple(i -> mult[i] * colstrides(bufsz)[i], N)
    return StridedView(rand(T, prod(bufsz)), sz, strides)
end

# Expand a group spec into concrete case instances (one per swept total).
function expand(spec, group)
    totals = haskey(spec, "totals") ? Int.(spec["totals"]) : Int[]
    insts = NamedTuple[]
    for c in spec["cases"]
        p = (Int.(c["p"])...,)
        isperm(p) || error("$group: `p = $(c["p"])` is not a permutation")
        N = length(p)
        min = haskey(c, "stride_in") ? (Int.(c["stride_in"])...,) : ntuple(one, N)
        mout = haskey(c, "stride_out") ? (Int.(c["stride_out"])...,) : ntuple(one, N)
        (length(min) == N == length(mout)) ||
            error("$group: `stride_in`/`stride_out` must have length $N")
        if haskey(c, "size")
            dims = (Int.(c["size"])...,)
            push!(insts, (; p, dims, min, mout, L = prod(dims), shape = nothing))
        elseif haskey(c, "shape")
            shape = (Float64.(c["shape"])...,)
            length(shape) == N || error("$group: `shape` must have length $N")
            isempty(totals) &&
                error("$group: case uses `shape` but the group has no `totals`")
            for total in totals
                push!(insts, (; p, dims = shapedims(shape, total), min, mout, L = total, shape))
            end
        else
            error("$group: each case needs a `size` or a `shape`")
        end
    end
    return insts
end

function casename(inst)
    s = "p=" * join(inst.p) * "_L=" * string(inst.L)
    inst.shape === nothing || (s *= "_shape=" * join(round.(Int, inst.shape), 'x'))
    any(!isone, inst.min) && (s *= "_sin=" * join(inst.min, 'x'))
    any(!isone, inst.mout) && (s *= "_sout=" * join(inst.mout, 'x'))
    return s
end

# Arrays are allocated (and touched) in the per-sample setup rather than at
# suite-construction time: the full grid would otherwise keep every array alive
# at once. `evals` is fixed so that tuning never re-runs the setup, and is > 1
# only for cases too short to time reliably in a single evaluation.
function addcase!(group::BenchmarkGroup, inst, ::Type{T}, nt; samples, seconds) where {T}
    L = prod(inst.dims)
    nt > 1 && L < MINTHREADLENGTH && return group
    dsz = ntuple(i -> inst.dims[inst.p[i]], length(inst.p))
    evals = max(1, (1 << 15) ÷ L)
    p, sz, min, mout = inst.p, inst.dims, inst.min, inst.mout
    group[casename(inst)] = @benchmarkable(
        permutedims!(dst, src, $p),
        setup = (
            Strided.set_num_threads($nt);
            src = make_view($T, $sz, $min);
            dst = make_view($T, $dsz, $mout)
        ),
        evals = evals,
        samples = samples,
        seconds = seconds,
    )
    return group
end

function addcopy!(group::BenchmarkGroup, L, ::Type{T}) where {T}
    evals = max(1, (1 << 15) ÷ L)
    group["L=$L"] = @benchmarkable(
        copyto!(dst, src),
        setup = (src = rand($T, $L); dst = Vector{$T}(undef, $L)),
        evals = evals,
        samples = 100,
        seconds = 5,
    )
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

const SUITE = let
    suite = BenchmarkGroup()
    permute = addgroup!(suite, "permutedims!")
    lengths = Set{Int}()
    for name in GROUPS
        haskey(CASEGROUPS, name) ||
            error("--groups: group `$name` not found in $(CONFIG["cases"])")
        spec = CASEGROUPS[name]
        insts = expand(spec, name)
        samples, seconds = spec["samples"], spec["seconds"]
        g = addgroup!(permute, name)
        for T in ELTYPES
            tg = addgroup!(g, "T=$T")
            for nt in NTHREADS
                ng = addgroup!(tg, "nthreads=$nt")
                for inst in insts
                    addcase!(ng, inst, T, nt; samples, seconds)
                    push!(lengths, inst.L)
                end
            end
        end
    end
    copies = addgroup!(suite, "copy")
    for T in ELTYPES
        tg = addgroup!(copies, "T=$T")
        for L in sort!(collect(lengths))
            addcopy!(tg, L, T)
        end
    end
    isempty(CONFIG["filter"]) ? suite : filtered(suite, CONFIG["filter"])
end
