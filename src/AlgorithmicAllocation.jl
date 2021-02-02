module AlgorithmicAllocation

using FillArrays
using IRTools
using LinearAlgebra
using MacroTools

using IRTools: @dynamo, IR, @code_ir, xcall, insertafter!, meta, blocks, varargs!
using IRTools.Inner: argnames!, update!
using MacroTools: prewalk, postwalk

function preallocate(::typeof(*), A::Matrix{<:Real}, B::VecOrMat{<:Real})
    C = A * B
    mem = deepcopy(C)
    function preallocated_mul(A::Matrix{<:Real}, B::VecOrMat{<:Real})
        return mul!(mem, A, B)
    end
    return C, preallocated_mul
end

function preallocate(::typeof(*), A::Matrix, B::Adjoint{<:Any, <:Matrix})
    C = A * B
    mem = deepcopy(C)
    function preallocated_mul(A::Matrix, B::Adjoint{<:Any, <:Matrix})
        return mul!(mem, A, B)
    end
    return C, preallocated_mul
end

function preallocate(::typeof(+), A::Array, B::Array)
    C = A + B
    mem = deepcopy(C)
    function preallocated_add(A::Array, B::Array)
        C .= A .+ B
        return C
    end
    return C, preallocated_add
end

function preallocate(::typeof(+), A::AbstractArray, B::Zeros)
    C = deepcopy(A)
    function preallocated_add_zero(A::Array, B::Zeros)
        C .= A
        return C
    end
    return A + B, preallocated_add_zero
end

preallocate(::Type{<:Adjoint}, A) = A', x -> x'

function preallocate(::typeof(Core.tuple), args...)
    return Core.tuple(args...), Core.tuple
end

# A really simple data structure that can be used with a pre-allocated function.
struct Preallocated{signature, T<:Tuple}
    preallocated_functions::T
end

function Preallocated{signature}(preallocated_functions::T) where {signature, T<:Tuple}
    return Preallocated{signature, T}(preallocated_functions)
end

function preallocated(signature, preallocated_functions::T) where {T}
    return Preallocated{signature, T}(preallocated_functions)
end

@generated function preallocate(f, args...)
    signature = Tuple{f, args...}
    metadata = meta(signature)
    ir = IR(metadata)
    preallocations = []

    # Modify all expressions to instead call preallocate.
    new_arg = IRTools.argument!(ir; at=1)
    for (x, st) in only(blocks(ir))
        isexpr(st.expr, :call) || continue

        # Insert a call to preallocate just before the function is evaluated.
        new_x = insert!(ir, x, xcall(Main, :preallocate, st.expr.args...))

        # Replace the original expression with the first element of the output of
        # preallocate, which is defined to be the same as just evaluating the function.
        ir[x] = xcall(:getindex, new_x, 1)

        # Insert an expression immediately after containing the preallocated function.
        new_preallocation = insertafter!(ir, x, xcall(:getindex, new_x, 2))

        # Store the location of the preallocated function for later.
        push!(preallocations, new_preallocation)
    end

    # Construct a `Preallocated` struct that runs the preallocated version of the function.
    preallocated_data = push!(ir, xcall(:tuple, preallocations...))
    new_allocation = push!(
        ir, xcall(Main, :preallocated, signature, preallocated_data),
    )

    # Set the function to return a tuple comprising the usual return value and the
    # Preallocated data structure.
    old_return = IRTools.returnvalue(only(IRTools.branches(ir)))
    new_return = push!(ir, xcall(:tuple, old_return, new_allocation))
    IRTools.return!(ir, new_return)

    # Hack the metadata to make everything. I don't understand this bit at all.
    argnames!(metadata, Symbol("#self#"), :f, :args)
    ir = varargs!(metadata, ir, 2)
    return update!(metadata.code, ir)
end

(p::Preallocated)(args...) = _preallocated(p::Preallocated, args...)

@generated function _preallocated(p::Preallocated{signature}, args...) where {signature}
    metadata = meta(signature)
    ir = IR(metadata)

    # Add an extra argument to the IR that is p.
    p_arg = IRTools.argument!(ir; at=2)

    # Pull the preallocated functions out of p.
    preallocated_functions = pushfirst!(ir, xcall(:getfield, p_arg, :(:preallocated_functions)))

    n = 0
    for (x, st) in only(blocks(ir))
        isexpr(st.expr, :call) || continue
        n += 1
        n == 1 && continue

        # Insert a call to preallocate just before the function is evaluated to get the
        # preallocated version of the function.
        new_x = insert!(ir, x, xcall(:getindex, preallocated_functions, n - 1))

        # preallocated_call = insert!(ir, x, xcall(:getindex, ))
        # Replace the original call with the pre-allocated call.
        ir[x] = xcall(new_x, st.expr.args[2:end]...)
    end

    # Again, hack the metadata to make everything. I don't understand this bit at all.
    argnames!(metadata, Symbol("#self#"), :p, :args)
    ir = varargs!(metadata, ir, 2)
    return update!(metadata.code, ir)
end

end
