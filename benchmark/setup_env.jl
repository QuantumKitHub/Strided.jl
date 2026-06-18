# Activate a benchmark environment that uses the *local* Strided checkout
# (the package living one directory up). Works unchanged inside any worktree.
import Pkg
Pkg.activate(@__DIR__)
let root = normpath(joinpath(@__DIR__, ".."))
    # `develop` is idempotent; re-pointing to the local path each run guarantees
    # we benchmark this worktree's source rather than a registered version.
    try
        Pkg.develop(Pkg.PackageSpec(path = root); io = devnull)
    catch
        Pkg.develop(Pkg.PackageSpec(path = root))
    end
    if !haskey(Pkg.project().dependencies, "BenchmarkTools")
        Pkg.add("BenchmarkTools"; io = devnull)
    end
end
Pkg.instantiate(; io = devnull)
