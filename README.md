<img src="https://github.com/QuantumKitHub/Strided.jl/blob/main/docs/src/assets/logo.svg"
width="150">

# Strided.jl

Strided array views with efficient (cache-friendly and multithreaded) manipulations

| **Build Status** | **Coverage** | **Quality assurance** | **Downloads** |
|:----------------:|:------------:|:---------------------:|:--------------|
| [![CI][ci-img]][ci-url] [![CI (Julia nightly)][ci-julia-nightly-img]][ci-julia-nightly-url] | [![Codecov][codecov-img]][codecov-url] | [![Aqua QA][aqua-img]][aqua-url] | [![Strided Downloads][downloads-img]][downloads-url] |

[github-img]: https://github.com/QuantumKitHub/Strided.jl/workflows/CI/badge.svg
[github-url]: https://github.com/QuantumKitHub/Strided.jl/actions?query=workflow%3ACI

[ci-img]: https://github.com/QuantumKitHub/Strided.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/QuantumKitHub/Strided.jl/actions?query=workflow%3ACI

[ci-julia-nightly-img]:
    https://github.com/QuantumKitHub/Strided.jl/workflows/CI%20(Julia%20nightly)/badge.svg
[ci-julia-nightly-url]:
    https://github.com/QuantumKitHub/Strided.jl/actions?query=workflow%3A%22CI+%28Julia+nightly%29%22

[codecov-img]: https://codecov.io/gh/QuantumKitHub/Strided.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/QuantumKitHub/Strided.jl

[aqua-img]: https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg
[aqua-url]: https://github.com/JuliaTesting/Aqua.jl

[downloads-img]: https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FStrided&query=total_requests&label=Downloads
[downloads-url]: http://juliapkgstats.com/pkg/Strided

A Julia package for working more efficiently with strided arrays, i.e. dense arrays whose
memory layout has a fixed stride along every dimension. Strided.jl does not make any
assumptions about the strides (such as stride 1 along first dimension, or monotonically
increasing strides) and provides multithreaded and cache friendly implementations for
mapping, reducing, broadcasting such arrays, as well as taking views, reshaping and
permuting dimensions. Most of these are simply accessible by annotating a block of standard
Julia code involving broadcasting and other array operations with the macro `@strided`.
Currently, Strided.jl only supports arrays in the main memory and does not provide
implementations for arrays on GPUs or other hardware accelerators.

# What's new

Strided.jl v2 reduces the complexity of the implementation. It discards of the
`UnsafeStridedView` type, which was pointer based and required to avoid allocations prior to
Julia v1.5 (because of [#14955](https://github.com/JuliaLang/julia/issues/14955)). The
associated `@unsafe_strided` macro has been deprecated.

The main structured type `StridedView` for representing a strided view over a contiguous
array (`DenseArray`) is now defined in a separate package
[StridedViews.jl](https://github.com/Jutho/StridedViews.jl). This definition is device
agnostic and can thus also be used in combination with dense GPU arrays. However, at the
moment, the methods implemented in Strided.jl are restricted to strided views over `Array`
data.

# Examples

Running Julia with a single thread

```julia
julia> using Strided

julia> using BenchmarkTools

julia> A = randn(4000,4000);

julia> B = similar(A);

julia> @btime $B .= ($A .+ $A') ./ 2;
  145.214 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= ($A .+ $A') ./ 2;
  56.189 ms (6 allocations: 352 bytes)

julia> A = randn(1000,1000);

julia> B = similar(A);

julia> @btime $B .= 3 .* $A';
  2.449 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= 3 .* $A';
  1.459 ms (5 allocations: 288 bytes)

julia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  22.493 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  22.240 ms (10 allocations: 480 bytes)

julia> A = randn(32,32,32,32);

julia> B = similar(A);

julia> @btime permutedims!($B, $A, (4,3,2,1));
  5.203 ms (2 allocations: 128 bytes)

julia> @btime @strided permutedims!($B, $A, (4,3,2,1));
  2.201 ms (4 allocations: 320 bytes)

julia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  21.863 ms (32 allocations: 32.00 MiB)

julia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  8.495 ms (9 allocations: 640 bytes)
```
And now with `export JULIA_NUM_THREADS = 4`
```julia
julia> using Strided

julia> using BenchmarkTools

julia> A = randn(4000,4000);

julia> B = similar(A);

julia> @btime $B .= ($A .+ $A') ./ 2;
  146.618 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= ($A .+ $A') ./ 2;
  30.355 ms (12 allocations: 912 bytes)

julia> A = randn(1000,1000);

julia> B = similar(A);

julia> @btime $B .= 3 .* $A';
  2.030 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= 3 .* $A';
  808.874 μs (11 allocations: 784 bytes)

julia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  21.971 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  5.811 ms (16 allocations: 1.05 KiB)

julia> A = randn(32,32,32,32);

julia> B = similar(A);

julia> @btime permutedims!($B, $A, (4,3,2,1));
  5.334 ms (2 allocations: 128 bytes)

julia> @btime @strided permutedims!($B, $A, (4,3,2,1));
  1.192 ms (10 allocations: 928 bytes)

julia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  22.465 ms (32 allocations: 32.00 MiB)

julia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  2.796 ms (15 allocations: 1.44 KiB)
```

# Design principles

## `StridedView`

Strided.jl is centered around the type `StridedView`, which provides a view into a parent
array of type `DenseArray` such that the resulting view is strided. The definition of this
type, together with the set of methods that create `StridedView` instances, and transform
them into eachother, are now implemented in
[StridedViews.jl](https://github.com/Jutho/StridedViews.jl). This package is device agnostic
and never actually operators on the data in a nontrivial manner.

## Broadcasting and `map(reduce)`

Whenever an expression only contains `StridedView`s and non-array objects (scalars),
overloaded methods for broadcasting and functions as `map(!)` and `mapreduce` are used that
exploit the known strided structure in order to evaluate the result in a more efficient way,
at least for sufficiently large arrays where the overhead of the extra preparatory work is
negligible. In particular, this involves choosing a blocking strategy and loop order that
aims to avoid cache misses. This matters in particular if some of the `StridedView`s
involved have strides which are not monotonously increasing, e.g. if `transpose`, `adjoint`
or `permutedims` has been applied. The fact that the latter also acts lazily (whereas it
creates a copy of the data in Julia base) can potentially provide a further speedup.

## The `@strided` macro annotation
Rather than manually wrapping every array in a `StridedView`, there is the macro annotation
`@strided some_expression`, which will wrap all `DenseArray`s appearing in `some_expression`
in a `StridedView`. Note that, because `StridedView`s behave lazily under indexing with
ranges, this acts similar to the `@views` macro in Julia Base, i.e. there is no need to use
a view.

The macro `@strided` acts as a contract, i.e. the user ensures that all array manipulations
in the following expressions will preserve the strided structure. Therefore, `reshape` and
`view` are are replaced by `sreshape` and `sview` respectively. As mentioned above,
`sreshape` will throw an error if the requested new shape is incompatible with preserving
the strided structure. The function `sview` is only defined for index arguments which are
ranges, `Int`s or `Colon` (`:`), and will thus also throw an error if indexed by anything
else.

## Multithreading support

The optimized methods in Strided.jl are implemented with support for multithreading. Thus,
if `Threads.nthreads() > 1` and the arrays involved are sufficiently large, performance can
be boosted even for plain arrays with a strictly sequential memory layout, provided that the
broadcast operation is compute bound and not memory bound (i.e. the broadcast function is
sufficienlty complex).

Strided.jl uses the @spawn threading infrastructure, and the number of tasks that will be
spawned is customizable via the function `Strided.set_num_threads(n)`, where `n` can be any
integer between 1 (no threading) and `Base.Threads.nthreads()`. This allows to spend only a
part of the Julia threads on multithreading, i.e. Strided will never spawn more than `n-1`
additional tasks. By default, `n = Base.Threads.nthreads()`, i.e. threading is enabled by
default. There are also convenience functions `Strided.enable_threads() =
Strided.set_num_threads(Threads.nthreads())` and `Strided.disable_threads() =
Strided.set_num_threads(1)`.

Furthermore, there is an experimental feature (disabled by default) to apply multithreading
for matrix multiplication using a divide-and-conquer strategy. It can be enabled via
`Strided.enable_threaded_mul()` (and similarly `Strided.disable_threaded_mul()` to revert to
the default setting). For matrices with a `LinearAlgebra.BlasFloat` element type (i.e. any
of `Float32`, `Float64`, `ComplexF32` or `ComplexF64`), this is typically not necessary as
BLAS is multithreaded by default. However, it can be beneficial to implement the
multithreading using Julia Tasks, which then run on Julia's threads as distributed by
Julia's scheduler. Hence, this feature should likely be used in combination with
`LinearAlgebra.BLAS.set_num_threads(1)`. Performance seems to be on par (within a few
percent margin) with the threading strategies of OpenBLAS and MKL. However, note that the
latter call also disables any multithreading used in LAPACK (e.g. `eigen`, `svd`, `qr`, ...)
and Strided.jl does not help with that.

## `StridedView` versus `StridedArray` and BLAS/LAPACK compatibility

`StridedArray` is a union type to denote arrays with a strided structure in Julia Base.
Because of its definition as a type union rather than an abstract type, it is impossible to
have user types be recognized as `StridedArray`. This is rather unfortunate, since
dispatching to BLAS and LAPACK routines is based on `StridedArray`. As a consequence,
`StridedView` will not fall back to BLAS or LAPACK by default. Currently, only matrix
multiplication is overloaded so as to fall back to BLAS (i.e. `gemm!`) if possible. In
general, one should not attempt use e.g. matrix factorizations or other lapack operations
within the `@strided` context. Support for this is on the TODO list. Some BLAS inspired
methods (`axpy!`, `axpby!`, scalar multiplication via `mul!`, `rmul!` or `lmul!`) are
however overloaded by relying on the optimized yet generic `map!` implementation.

`StridedView`s can currently only be created with certainty from `DenseArray` (typically
just `Array` in Julia Base). For `Base.SubArray` or `Base.ReshapedArray` instances, the
`StridedView` constructor will first act on the underlying parent array, and then try to
mimic the corresponding view or reshape operation using `sview` and `sreshape`. These,
however, are more limited then their Base counterparts (because they need to guarantee that
the result still has a strided memory layout with respect to the new dimensions), so an
error can result. However, this approach can also succeed in creating `StridedView` wrappers
around combinations of `view` and `reshape` that are not recognised as `Base.StridedArray`.
For example, `reshape(view(randn(40,40), 1:36, 1:20), (6,6,5,4))` is not a
`Base.StridedArrray`, and indeed, it cannot statically be inferred to be strided, from only
knowing the argument types provided to `view` and `reshape`. For example, the similarly
looking `reshape(view(randn(40,40), 1:36, 1:20), (6,3,10,4))` is not strided. The
`StridedView` constructor will try to act on both, and yield a runtime error in the second
case. Note that `Base.ReinterpretArray` is currently not supported.

Note again that, unlike `StridedArray`s, `StridedView`s behave lazily (i.e. still produce a
view on the same parent array) under `permutedims` and regular indexing with ranges.

# Planned features / wish list

*   Support for `GPUArray`s with dedicated GPU kernels?
