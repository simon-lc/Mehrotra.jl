include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))


@testset "complementarity decoupling: random NCP (non negative cone)" begin
    # without decoupling
    options = Mehrotra.Options(
        verbose=false,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )

    solver = random_lcp(; num_primals=6, num_cone=12,
        cone_type=:non_negative_cone,
        options=options,
        seed=1,
        )

    Mehrotra.solve!(solver)
    iterations = solver.trace.iterations
    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance

    # with decoupling
    solver.options.verbose = true
    solver.options.complementarity_decoupling = true
    decoupling_iterations = solver.trace.iterations
    Mehrotra.solve!(solver)

    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance
    @test decoupling_iterations == iterations
end

@testset "complementarity decoupling: random NCP (second order cone)" begin
    # without decoupling
    options = Mehrotra.Options(
        verbose=false,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )

    solver = random_lcp(; num_primals=6, num_cone=12,
        cone_type=:second_order_cone,
        options=options,
        seed=1,
        )

    Mehrotra.solve!(solver)
    iterations = solver.trace.iterations
    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance

    # with decoupling
    # solver.options.verbose = true
    solver.options.complementarity_decoupling = true
    decoupling_iterations = solver.trace.iterations
    Mehrotra.solve!(solver)

    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance
    @test decoupling_iterations == iterations
end
