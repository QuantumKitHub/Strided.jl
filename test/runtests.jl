using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView
using CUDA
using Aqua
using CUDA: Adapt

Random.seed!(1234)

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    println("Base.Threads.nthreads() =  $(Base.Threads.nthreads())")

    println("Running tests single-threaded:")
    Strided.disable_threads()
    include("othertests.jl")
    include("blasmultests.jl")

    println("Running tests multi-threaded:")
    Strided.enable_threads()
    Strided.set_num_threads(Base.Threads.nthreads() + 1)
    include("othertests.jl")
    include("blasmultests.jl")

    Strided.enable_threaded_mul()
    include("blasmultests.jl")
    Strided.disable_threaded_mul()

    Aqua.test_all(Strided; piracies=false)
end

if CUDA.functional()
    include("cuda.jl")
end
