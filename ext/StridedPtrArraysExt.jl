module StridedPtrArraysExt

using Strided
using PtrArrays

Strided._normalizeparent(A::PtrArray) = PtrArray(A.ptr, length(A))

end
