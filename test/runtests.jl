using Test
using LinearAlgebra
using Random
using Strided
using Strided: StridedView
using Aqua
using JLArrays, AMDGPU, CUDACore, cuRAND, GPUArrays

Random.seed!(1234)

is_buildkite = get(ENV, "BUILDKITE", "false") == "true"

if !is_buildkite
    include("jlarrays.jl")
    @testset "JLArray GPU mapreduce" begin
        include("mapreduce_gpu.jl")
    end
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

    Aqua.test_all(Strided; piracies = false)
end

if CUDACore.functional()
    include("cuda.jl")
end

if AMDGPU.functional()
    include("amd.jl")
end
