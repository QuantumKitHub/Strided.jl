module Strided

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims
const SliceIndex = Union{RangeIndex, Colon}

using LinearAlgebra

using TupleTools
# using TupleTools: StaticLength

include("auxiliary.jl")
include("stridedview.jl")

# re-export?
export StridedView, @strided, sreshape, sview, isstrided

const _NTHREADS = Ref(1)
get_num_threads() = _NTHREADS[]

function set_num_threads(n::Int)
    N = Base.Threads.nthreads()
    if n > N
        n = N
        _set_num_threads_warn(n)
    end
    return _NTHREADS[] = n
end
@noinline function _set_num_threads_warn(n)
    return @warn "Maximal number of threads limited by number of Julia threads,
            setting number of threads equal to Threads.nthreads() = $n"
end

enable_threads() = set_num_threads(Base.Threads.nthreads())
disable_threads() = set_num_threads(1)

const _use_threaded_mul = Ref(false)
use_threaded_mul() = _use_threaded_mul[]

function disable_threaded_mul()
    _use_threaded_mul[] = false
    return
end

function enable_threaded_mul()
    _use_threaded_mul[] = true
    return
end

function __init__()
    return set_num_threads(Base.Threads.nthreads())
end

include("linalg.jl")
include("mapreduce.jl")
include("broadcast.jl")
include("macros.jl")
include("convert.jl")

end
