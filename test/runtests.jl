using AlgorithmicAllocation
using BenchmarkTools
using IRTools
using LinearAlgebra
using Stheno
using TemporalGPs
using Test
using Zygote

using AlgorithmicAllocation: preallocate, preallocated, _preallocated
using LinearAlgebra: Adjoint

# Assume for now that f is immutable.
function test_preallocate(f, args...)
    y, mem, pf! = preallocate(f, args...)
    @test y == f(args...)
    @test y == pf!(mem, args...)
end

@testset "AlgorithmicAllocation.jl" begin

    # Test intrinsics. These necessarily require specialised preallocate methods because
    # they don't have any IR to overload.
    test_preallocate(Core.arraysize, randn(5, 4), 1)
    test_preallocate(Core.arraysize, randn(5, 4), 2)

    # Test allocating functions. These are the methods that I think are probably actually
    # necessary to make this work well.
    test_preallocate(Array{Union{Float64, Nothing}, 2}, nothing, 2, 3)

    # Test functions that shouldn't really need custom preallocate functions, but do because
    # of shortcomings in the present pre-allocation algorithm.
    test_preallocate(eltype, Array{Float64, 2})
    test_preallocate(LinearAlgebra.MulAddMul, true, false)
    test_preallocate(LinearAlgebra.MulAddMul, 0, 1)
    test_preallocate(fill!, Array{Union{Float64, Nothing}}(undef, 3, 2), nothing)

    test_preallocate(*, randn(5, 5), randn(5, 5))
    test_preallocate(*, randn(5, 5), randn(5, 5)')
    test_preallocate(*, randn(5, 5), randn(5))
    test_preallocate(+, randn(5, 4), randn(5, 4))
    test_preallocate(+, randn(4), randn(4))
    test_preallocate(Adjoint, randn(5, 5))
    test_preallocate(Core.tuple, randn(5, 4), randn(3))

    A = randn(5, 5);
    B = randn(5, 5);

    function foo(A, B)
        return A * B
    end

    C, mem, pre_foo! = preallocate(foo, A, B)
 
    @benchmark foo($A, $B)
    @benchmark $pre_foo!($mem, $A, $B)

    function bar(A, B)
        C = foo(A, B)
        D = foo(C, B)
        return C * D
    end

    bar(A, B)

    @benchmark bar($A, $B)

    _, mem, pre_bar! = preallocate(bar, A, B)
    @benchmark $pre_bar!($mem, $A, $B)

    f = to_sde(GP(Matern52(), GPC()));
    x = RegularSpacing(0.0, 1.0, 10);
    s = 0.1;

    lgssm = TemporalGPs.build_lgssm(f(x, s));

    model_1 = lgssm.model[1];
    x0 = lgssm.model.gmm.x0;

    args = (x0.m, x0.P, model_1.gmm.A, model_1.gmm.a, model_1.gmm.Q);
    @benchmark TemporalGPs.predict($args...)

    # Well, this is nearly what I wanted :shrug:.
    _, mem, pre_predict! = preallocate(TemporalGPs.predict, args...);
    @benchmark $pre_predict!($mem, $args...)

    @code_warntype _preallocated(pre_predict!, mem, args...)

    @code_warntype Zygote._pullback(Zygote.Context(), TemporalGPs.predict, args...)

    _, mem, _pullback! = preallocate(Zygote._pullback, Zygote.Context(), TemporalGPs.predict, args...)

    # Check that a simple closure works as planned. Zygote is only a bit more compilcated
    # than this really.
    function foo(x)
        function bar(y)
            return x + y
        end
        return bar
    end

    __bar, mem, pa! = preallocate(foo, randn(5, 5))

    # Can't presently hack closures. We could extend to be able to hack closures by simply
    # keeping hold of the original function in the Preallocated object, and making it
    # available to the pre-allocated version of the function. This is straightforward in
    # principle, but it's not clear that it's entirely straightforward in practice.
    _, mem_bar, pa_bar! = preallocate(__bar, randn(5, 5))

    _, mem, _pullback! = preallocate(Zygote._pullback, Zygote.Context(), *, A, B)



    @inline function AutomaticAllocation.preallocate(::typeof(Zygote.chain_rrule), f, args...)
        (y, back), mem, rrule_pa! = preallocate(rrule, f, args...)
        function 
        return (y, ZBack(back)), mem, 
    end

    @code_warntype preallocate(Zygote._pullback, Zygote.Context(), *, A, B)

    # This works, and is what goes on internally.
    Core._apply_iterate(Base.iterate, Zygote.chain_rrule, Core.tuple(*), (A, B))

    # # This makes things play nicely with the Zygote forwards-pass when a ChainRule is hit.
    # function AlgorithmicAllocation.preallocate(
    #     ::typeof(Core._apply_iterate),
    #     ::typeof(Base.iterate),
    #     ::typeof(Zygote.chain_rrule),
    #     ft::Tuple,
    #     args,
    # )
    #     out, mem, pa! = preallocate(Zygote.chain_rrule, ft[1], args...)
    #     return out, mem, (mem, _, _, ft, args) -> pa!(mem, ft[1], args...)
    # end

    args = (x0.m, x0.P, model_1.gmm.A, model_1.gmm.a, model_1.gmm.Q);
    @benchmark TemporalGPs.predict($args...)

    # Works like a charm!
    _, mem, pre_predict! = preallocate(TemporalGPs.predict, args...);
    @benchmark $pre_predict!($mem, $args...)

    # Attempt to do the forwards-pass.
    _, mem, _pullback! = preallocate(
        Zygote._pullback, Zygote.Context(), TemporalGPs.predict, args...,
    );
    _pullback!(mem, Zygote.Context(), TemporalGPs.predict, args...)

    @benchmark $_pullback!($mem, $args...)

end
