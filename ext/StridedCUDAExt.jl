module StridedCUDAExt

using Strided, CUDA
import Strided: _mapreduce_fuse!

ALL_FS = Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}

function Base.copy!(dst::StridedView{TD,ND,TAD,FD}, src::StridedView{TS,NS,TAS,FS}) where {TD, ND, TAD <: CuArray{TD}, FD<:ALL_FS, TS, NS, TAS <: CuArray{TS}, FS<:ALL_FS}
    all_dst_inds = map(ix->Strided.StridedViews._computeind(Tuple(ix), dst.strides), CartesianIndices(size(dst)))
    viewed_dst = view(parent(dst), all_dst_inds)
    all_src_inds = map(ix->Strided.StridedViews._computeind(Tuple(ix), src.strides), CartesianIndices(size(src)))
    viewed_src = view(parent(src), all_src_inds)
    return map!(identity, viewed_dst, viewed_src)
end

end
