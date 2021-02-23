module AlgorithmicAllocation

using FillArrays
using IRTools
using LinearAlgebra
using MacroTools

using IRTools: @dynamo, IR, @code_ir, xcall, insertafter!, meta, blocks, varargs!
using IRTools.Inner: argnames!, update!
using MacroTools: prewalk, postwalk

using LinearAlgebra: MulAddMul, gemm_wrapper!

# # Use very tight type constraints for custom operations on matrices as we should expect to
# # automatically derive rules in most cases.
# function preallocate(::typeof(*), A::Matrix{<:Real}, B::VecOrMat{<:Real})
#     C = A * B
#     return C, deepcopy(C), mul!
# end

# function preallocate(::typeof(*), A::Matrix{<:Real}, B::Adjoint{<:Real, <:Matrix})
#     C = A * B
#     return C, deepcopy(C), mul!
# end

# function preallocate(::typeof(+), A::Array, B::Array)
#     function preallocated_add!(C::Array, A::Array, B::Array)
#         C .= A .+ B
#         return C
#     end
#     C = A + B
#     return C, deepcopy(C), preallocated_add!
# end

# function preallocate(::typeof(+), A::AbstractArray, B::Zeros)
#     function preallocated_add_zero!(C::Array, A::Array, B::Zeros)
#         C .= A
#         return C
#     end
#     C = A + B
#     return C, deepcopy(C), preallocated_add_zero!
# end

# preallocate(::Type{<:Adjoint}, A) = A', nothing, (_, x) -> x'

function preallocate(::Type{<:Array{T, N}}, ::UndefInitializer, dims...) where {T, N}
    out = Array{T, N}(undef, dims...)
    mem = Array{T, N}(undef, dims...)
    return out, mem, (mem, args...) -> mem
end

function preallocate(::typeof(eltype), T::Type{<:AbstractArray})
    return eltype(T), nothing, (_, T) -> eltype(T)
end

function preallocate(::typeof(MulAddMul), alpha, beta)
    return MulAddMul(alpha, beta), nothing, (_, alpha, beta) -> MulAddMul(alpha, beta)
end

function preallocate(::typeof(gemm_wrapper!), args...)
    return gemm_wrapper!(args...), nothing, (_, args...) -> gemm_wrapper!(args...)
end

function preallocate(::typeof(fill!), args...)
    return fill!(args...), nothing, (_, args...) -> fill!(args...)
end

#
# Internal stuff that I need some equivalent to @nograd for. All of these functions are
# non-allocating anyway, so it would be nice to have something that can specify that, and
# make life simple for the compiler by indicating that a given function should be left
# unchanged in the IR.
# All of these things return `nothing` when you ask for their IR.
#

@inline _tuple_pa!(::Nothing, args::Vararg{Any, N}) where {N} = Core.tuple(args...)

preallocate(::typeof(Core.tuple), args...) = Core.tuple(args...), nothing, _tuple_pa!

function preallocate(::typeof(Core._apply_iterate), iter, f, args::Vararg{Any, N}) where {N}
    out, mem, pa! = Core._apply_iterate(iter, preallocate, (f, ), args...)
    function _apply_iterate_pa!(mem, a, b, args_new::Vararg{Any, N}) where {N}
        return Core._apply_iterate(iter, pa!, (mem,), args_new...)
    end
    return out, mem, _apply_iterate_pa!
end

# @inline function Cassette.overdub(ctx::$Ctx, ::typeof(Core._apply_iterate), iter, f, args...)
#     # XXX: Overdubbing the iterate function breaks things
#     # overdubbed_iter = (args...) -> Core._apply(overdub, (ctx, iter), args...)
#     Core._apply_iterate(iter, overdub, (ctx, f), args...)
# end

# This is actually a core rule -- we assume that, by default, the creation of types doesn't
# allocate. So for any types that _do_ allocate, we must override their creation directly.
# Whether this means overriding this function, or something more high-level, is neither here
# nor there.
function preallocate(::typeof(Core.apply_type), args...)
    return Core.apply_type(args...), nothing, (_, args...) -> Core.apply_type(args...)
end

function preallocate(::typeof(fieldtype), args...)
    return fieldtype(args...), nothing, (_, args...) -> fieldtype(args...)
end

function preallocate(::typeof(typeof), x)
    return typeof(x), nothing, (_, x) -> typeof(x)
end

function preallocate(::typeof(getfield), args...)
    function getfield_pa!(_, args::Vararg{Any, N}) where {N}
        return getfield(args...)
    end
    return getfield(args...), nothing, getfield_pa!
end

function preallocate(::typeof(Base.indexed_iterate), args...)
    return Base.indexed_iterate(args...), nothing, (_, args...) -> Base.indexed_iterate(args...)
end

function preallocate(::typeof(Core.arraysize), args...)
    return Core.arraysize(args...), nothing, (_, args...) -> Core.arraysize(args...)
end

function preallocate(::typeof(Core.Compiler.return_type), args...)
    return Core.Compiler.return_type(args...), nothing, (_, args...) -> Core.Compiler.return_type(args...)
end

# A really simple data structure that can be used with a pre-allocated function.
struct Preallocated{signature, T<:Tuple}
    preallocated_functions::T
end

Base.show(io::IO, p::Preallocated{s}) where {s} = Base.show(io, "Preallocated{$s}")

function Preallocated{signature}(preallocated_functions::T) where {signature, T<:Tuple}
    return Preallocated{signature, T}(preallocated_functions)
end

function preallocated(signature, preallocated_functions::T) where {T}
    return Preallocated{signature, T}(preallocated_functions)
end

@generated function preallocate(f, args...)
    signature = Tuple{f, args...}
    metadata = meta(signature)

    # If there isn't any metadata, it probably means that we've hit an intrinsic, and should
    # bail out.
    if metadata === nothing
        throw(error("Attempted to get IR for $f with args $args."))
    end

    ir = IR(metadata)
    prepared_memories = []
    preallocations = []

    # Ensure that the single-block assumption isn't violated.
    if length(blocks(ir)) > 1
        throw(error("More than one block found in $f with args $args."))
    end

    # Modify all expressions to instead call preallocate.
    new_arg = IRTools.argument!(ir; at=1)
    for (x, st) in only(blocks(ir))
        isexpr(st.expr, :call) || continue

        # Insert a call to preallocate just before the function is evaluated.
        new_x = insert!(ir, x, xcall(Main, :preallocate, st.expr.args...))

        # Replace the original expression with the first element of the output of
        # preallocate, which is defined to be the same as just evaluating the function.
        ir[x] = xcall(:getindex, new_x, 1)

        # Insert an expression containing the prepared memory.
        new_prepared_memory = insertafter!(ir, x, xcall(:getindex, new_x, 2))

        # Store the location of the prepared memory for later.
        push!(prepared_memories, new_prepared_memory)

        # Insert an expression immediately after containing the preallocated function.
        new_preallocation = insertafter!(ir, new_prepared_memory, xcall(:getindex, new_x, 3))

        # Store the location of the preallocated function for later.
        push!(preallocations, new_preallocation)
    end

    # Construct a Tuple containing the prepared memory.
    prepared_memory = push!(ir, xcall(:tuple, prepared_memories...))

    # Construct a `Preallocated` struct that runs the preallocated version of the function.
    preallocated_data = push!(ir, xcall(:tuple, preallocations...))
    new_allocation = push!(
        ir, xcall(Main, :preallocated, signature, preallocated_data),
    )

    # Set the function to return a tuple comprising the usual return value, the prepared
    # memory, and the Preallocated data structure.
    old_return = IRTools.returnvalue(only(IRTools.branches(ir)))
    new_return = push!(ir, xcall(:tuple, old_return, prepared_memory, new_allocation))
    IRTools.return!(ir, new_return)

    # Hack the metadata to make everything. I don't understand this bit at all.
    argnames!(metadata, Symbol("#self#"), :f, :args)
    ir = varargs!(metadata, ir, 2)
    return update!(metadata.code, ir)
end

(p::Preallocated)(mem, args...) = _preallocated(p, mem, args...)

@generated function _preallocated(p::Preallocated{signature}, mem, args...) where {signature}
    metadata = meta(signature)
    ir = IR(metadata)

    # Add an extra argument to the IR that is p.
    p_arg = IRTools.argument!(ir; at=2)

    # Add an extra argument to the IR that is mem.
    mem_arg = IRTools.argument!(ir; at=3)

    # Pull the preallocated functions out of p.
    preallocated_functions = pushfirst!(ir, xcall(:getfield, p_arg, :(:preallocated_functions)))

    n = 0
    for (x, st) in only(blocks(ir))
        isexpr(st.expr, :call) || continue
        n += 1
        n == 1 && continue

        # Insert a call to getindex just before the function is evaluated to get the
        # preallocated version of the function.
        new_x = insert!(ir, x, xcall(:getindex, preallocated_functions, n - 1))

        # Insert a call to getindex just before the function is evaluated to get the
        # appropriate bit of memory.
        mem_n = insert!(ir, x, xcall(:getindex, mem_arg, n - 1))

        # preallocated_call = insert!(ir, x, xcall(:getindex, ))
        # Replace the original call with the pre-allocated call.
        ir[x] = xcall(new_x, mem_n, st.expr.args[2:end]...)
    end

    # Again, hack the metadata to make everything. I don't understand this bit at all.
    argnames!(metadata, Symbol("#self#"), :p, :mem, :args)
    ir = varargs!(metadata, ir, 3)
    return update!(metadata.code, ir)
end

# export _allocates, __allocates

# @generated function _allocates(f, args...)
#     signature = Tuple{f, args...}
#     metadata = meta(signature)
#     ir = IR(metadata)

#     Core.println("Type of blocks is")
#     Core.println(typeof(only(blocks(ir))))
#     for (x, st) in only(blocks(ir))
#         # Core.println(n)
#         Core.println(x)
#         Core.println(st)

#         if isexpr(st.expr, :call)
#             Core.println(typeof(st.expr.args[1].name))
#             ref = st.expr.args[1]
#             Core.println(typeof(getfield(ref.mod, ref.name)))
#         end
#         isexpr(st.expr, :call) || continue
#     end


#     # Again, hack the metadata to make everything. I don't understand this bit at all.
#     argnames!(metadata, Symbol("#self#"), :f, :args)
#     ir = varargs!(metadata, ir, 2)
#     return update!(metadata.code, ir)
# end

# function __allocates(signature)
#     metadata = meta(signature)
#     ir = IR(metadata)

#     Core.println("Type of blocks is")
#     Core.println(typeof(only(blocks(ir))))
#     for (x, st) in only(blocks(ir))
#         # Core.println(n)
#         Core.println(x)
#         Core.println(st)

#         if isexpr(st.expr, :call)
#             Core.println(typeof(st.expr.args[1].name))
#             ref = st.expr.args[1]
#             Core.println(typeof(getfield(ref.mod, ref.name)))
#             will_allocate = _allocates()
#         end
#         isexpr(st.expr, :call) || continue
#     end

#     return ir
# end

end
