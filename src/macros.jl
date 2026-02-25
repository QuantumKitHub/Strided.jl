macro strided(ex)
    ex = macroexpand(__module__, ex)
    return esc(_strided(ex))
end

function _strided(ex::Expr)
    if ex.head == :. && ex.args[2] isa QuoteNode # field access: wrap whole expression in maybestrided
        return Expr(:call, :(Strided.maybestrided), ex)
    elseif ex.head == :call && ex.args[1] isa Symbol
        if ex.args[1] == :reshape
            return Expr(:call, :(Strided.sreshape), map(_strided, ex.args[2:end])...)
        elseif ex.args[1] == :view
            return Expr(:call, :(Strided.sview), map(_strided, ex.args[2:end])...)
        else
            return Expr(:call, ex.args[1], map(_strided, ex.args[2:end])...)
        end
    elseif (ex.head == :(=) || ex.head == :(kw)) && ex.args[1] isa Symbol
        return Expr(
            ex.head, ex.args[1],
            Expr(:call, :(Strided.maybeunstrided), _strided(ex.args[2]))
        )
    elseif (ex.head == :(->))
        return Expr(
            ex.head, ex.args[1],
            Expr(:call, :(Strided.maybeunstrided), _strided(ex.args[2]))
        )
    else
        return Expr(ex.head, map(_strided, ex.args)...)
    end
end
const exclusionlist = Symbol[:(:)]
_strided(ex::Symbol) = ex in exclusionlist ? ex : Expr(:call, :(Strided.maybestrided), ex)
_strided(ex) = ex

maybestrided(A::StridedView) = A
maybestrided(A::AbstractArray) = StridedView(A)
maybestrided(A::Tuple) = maybestrided.(A)
maybestrided(A) = A
function maybeunstrided(A::StridedView)
    Ap = A.parent
    if size(A) == size(Ap) && strides(A) == strides(Ap) && offset(A) == 0 &&
            A.op == identity
        return Ap
    else
        return reshape(copy(A).parent, size(A))
    end
end
maybeunstrided(A::Tuple) = maybeunstrided.(A)
maybeunstrided(A) = A

# TODO: deprecate
macro unsafe_strided(args...)
    syms = args[1:(end - 1)]
    ex = macroexpand(__module__, args[end]) #_strided(args[end])
    all(isa(s, Symbol) for s in syms) ||
        error("The first arguments to `@unsafe_strided` must be variable names")

    ex = Expr(:let, Expr(:block, [:($s = Strided.StridedView($s)) for s in syms]...), ex)
    warnex = :(
        Base.depwarn(
            "`@unsafe_strided A B C ... ex` is deprecated, use `@strided ex` instead.",
            Symbol("@unsafe_strided"); force = true
        )
    )
    return esc(Expr(:block, warnex, ex))
end

export @unsafe_strided
