include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))

################################################################################
# non negative cone
################################################################################
@testset "lcp: non negative cone" begin
    Random.seed!(0)

    num_primals = 10
    num_cone = 10
    num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    As = rand(num_primals, num_primals)
    A = As' * As
    b = rand(num_primals)
    Cs = rand(num_cone, num_cone)
    C = Cs * Cs'
    d = rand(num_cone)
    parameters = [vec(A); b; vec(C); d]

    solver = Mehrotra.Solver(lcp_non_negative_cone_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options228(
            verbose=false,
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            )
        )

    Mehrotra.solve!(solver)
    @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
    @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
end
