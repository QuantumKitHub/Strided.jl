module StridedGPUArraysExt

using Strided, GPUArrays, LinearAlgebra
using GPUArrays: Adapt, KernelAbstractions
using GPUArrays.KernelAbstractions: @kernel, @index

ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

KernelAbstractions.get_backend(sv::StridedView{T, N, TA}) where {T, N, TA <: AnyGPUArray{T}} = KernelAbstractions.get_backend(parent(sv))

function Base.Broadcast.BroadcastStyle(gpu_sv::StridedView{T, N, TA}) where {T, N, TA <: AnyGPUArray{T}}
    raw_style = Base.Broadcast.BroadcastStyle(TA)
    return typeof(raw_style)(Val(N)) # sets the dimensionality correctly
end

function Base.copy!(dst::AbstractArray{TD, ND}, src::StridedView{TS, NS, TAS, FS}) where {TD <: Number, ND, TS <: Number, NS, TAS <: AbstractGPUArray{TS}, FS <: ALL_FS}
    bc_style = Base.Broadcast.BroadcastStyle(TAS)
    bc = Base.Broadcast.Broadcasted(bc_style, identity, (src,), axes(dst))
    GPUArrays._copyto!(dst, bc)
    return dst
end

# lifted from GPUArrays.jl
function Base.fill!(A::StridedView{T, N, TA, F}, x) where {T, N, TA <: AbstractGPUArray{T}, F <: ALL_FS}
    isempty(A) && return A
    @kernel function fill_kernel!(a, val)
        idx = @index(Global, Cartesian)
        @inbounds a[idx] = val
    end
    # ndims check for 0D support
    kernel = fill_kernel!(KernelAbstractions.get_backend(A))
    f_x = F <: Union{typeof(conj), typeof(adjoint)} ? conj(x) : x
    kernel(A, f_x; ndrange = size(A))
    return A
end

# kernel-based variant for copying between wrapped GPU arrays
@kernel function linear_copy_kernel!(dest, dstart, src, sstart, n)
    i = @index(Global, Linear)
    if i <= n
        @inbounds dest[dstart+i-1] = src[sstart+i-1]
    end
end

function Base.copyto!(dest::StridedView{TD, ND, TAD, FD}, dstart::Integer,
                      src::StridedView{TS, NS, TAS, FS}, sstart::Integer, n::Integer) where {TD, TS, ND, NS, TAD <: AbstractGPUArray{TD}, TAS <: AbstractGPUArray{TS}, FD, FS}
    n == 0 && return dest
    n < 0 && throw(ArgumentError(string("tried to copy n=", n, " elements, but n should be nonnegative")))
    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
    (checkbounds(Bool, destinds, dstart) && checkbounds(Bool, destinds, dstart+n-1)) || throw(BoundsError(dest, dstart:dstart+n-1))
    (checkbounds(Bool, srcinds, sstart)  && checkbounds(Bool, srcinds, sstart+n-1))  || throw(BoundsError(src,  sstart:sstart+n-1))
    kernel = linear_copy_kernel!(KernelAbstractions.get_backend(dest))
    kernel(dest, dstart, src, sstart, n; ndrange=n)
    return dest
end
Base.copyto!(dest::StridedView{TD, ND, TAD, FD}, src::StridedView{TS, NS, TAS, FS}) where {TD, TS, ND, NS, TAD <: AbstractGPUArray{TD}, TAS <: AbstractGPUArray{TS}, FD, FS} = copyto!(dest, 1, src, 1, length(src))


function LinearAlgebra.mul!(
        C::StridedView{TC, 2, <:AnyGPUArray{TC}},
        A::StridedView{TA, 2, <:AnyGPUArray{TA}},
        B::StridedView{TB, 2, <:AnyGPUArray{TB}},
        α::Number, β::Number
    ) where {TC, TA, TB}
    return GPUArrays.generic_matmatmul!(C, A, B, α, β)
end

end
