using AlgorithmicAllocation
using BenchmarkTools
using IRTools
using LinearAlgebra
using Stheno
using TemporalGPs
using Test

using AlgorithmicAllocation: preallocate, preallocated, _preallocated
using LinearAlgebra: Adjoint

# Assume for now that f is immutable.
function test_preallocate(f, args...)
    y, pf = preallocate(f, args...)
    @test y == f(args...)
    @test deepcopy(y) == pf(args...)
end

@testset "AlgorithmicAllocation.jl" begin

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

    C, pre_foo = preallocate(foo, A, B)
 
    C, pf = preallocate(*, A, B);
    @benchmark $pf($A, $B)
    @code_warntype pf(A, B)

    @code_ir pf(A, B)
    @code_warntype pf(A, B)

    @benchmark $pf($A, $B)

    @benchmark foo($A, $B)

    function bar(A, B)
        C = foo(A, B)
        D = foo(C, B)
        return C * D
    end

    bar(A, B)

    @benchmark bar($A, $B)

    _, pre_bar = preallocate(bar, A, B)
    @benchmark $pre_bar($A, $B)

    f = to_sde(GP(Matern52(), GPC()));
    x = RegularSpacing(0.0, 1.0, 10);
    s = 0.1;

    lgssm = TemporalGPs.build_lgssm(f(x, s));

    model_1 = lgssm.model[1];
    x0 = lgssm.model.gmm.x0;

    args = (x0.m, x0.P, model_1.gmm.A, model_1.gmm.a, model_1.gmm.Q);
    @benchmark TemporalGPs.predict($args...)

    # Well, this is nearly what I wanted :shrug:.
    _, pre_predict = preallocate(TemporalGPs.predict, args...);
    @benchmark $pre_predict($args...)
    @benchmark deepcopy($pre_predict($args...))

    @code_warntype _preallocated(pre_predict.preallocated_functions, args...)

end
