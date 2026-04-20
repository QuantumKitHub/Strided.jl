using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView
using Aqua
using Adapt, GPUArrays
using JLArrays
using AMDGPU
using CUDACore, cuRAND
using Metal

Random.seed!(1234)

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    println("Base.Threads.nthreads() =  $(Base.Threads.nthreads())")

    println("Running tests single-threaded:")
    Strided.disable_threads()
    include("othertests.jl")
    include("blasmultests.jl")

    if Base.Threads.nthreads() > 1
        println("Running tests multi-threaded:")
        Strided.enable_threads()
        Strided.set_num_threads(Base.Threads.nthreads() + 1)
        include("othertests.jl")
        include("blasmultests.jl")

        Strided.enable_threaded_mul()
        include("blasmultests.jl")
        Strided.disable_threaded_mul()
    else
        @warn "Cannot run multi-threaded tests"
    end

    Aqua.test_all(Strided; piracies = false)
end

include("gpu.jl")
