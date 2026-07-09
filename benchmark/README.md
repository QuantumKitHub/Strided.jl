# Strided.jl benchmarks

This directory contains a [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) suite (`benchmarks.jl`, defining `SUITE`) for the `permutedims!` machinery, in the format expected by [AirSpeedVelocity.jl](https://github.com/MilesCranmer/AirSpeedVelocity.jl) and PkgBenchmark.jl.
See the header of `benchmarks.jl` for the case selection.

## Running manually

`benchmarks.jl` only defines `SUITE`, but its contents are configurable through command-line arguments, which can be passed after a `--` separator:

```bash
julia --project=benchmark benchmark/benchmarks.jl --help
julia --project=benchmark -t 8 \
    -e 'include("benchmark/benchmarks.jl"); display(run(SUITE; verbose=true))' \
    -- -g small -T Float64 -f sz=1000
```

The first use requires instantiating the environment:

```bash
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Multithreaded variants are generated only when Julia is started with more than one thread (`-t` / `JULIA_NUM_THREADS`).
The full unfiltered grid (57 HPTT cases + 21 small cases + 42 unaligned cases, × 2 element types × 2 thread counts) takes on the order of an hour.

## Comparing revisions with AirSpeedVelocity

```bash
julia -e 'using Pkg; Pkg.add("AirSpeedVelocity")'  # once, in the global env

export JULIA_NUM_THREADS=8
benchpkg Strided --rev=v2.6.1,main,dirty --path=. --bench-on=main --filter=T=Float64
```

from the repository root.
`benchpkgtable` / `benchpkgplot` format the results.
AirSpeedVelocity includes the script with empty `ARGS`, so the default (full) configuration applies; restrict it with `--filter`, which matches substrings of the benchmark names (element type, thread count, permutation, and size are all part of the name).

Note that `--bench-on` requires a revision that already contains this suite; to benchmark with an uncommitted version of the script, pass it explicitly along with its non-standard dependencies, e.g. `--script=benchmark/benchmarks.jl --add=ArgParse`.
The `--output-dir` must exist beforehand, and the current AirSpeedVelocity version (0.6.5) crashes on `--bench-on=dirty`, so prefer a committed revision there.
