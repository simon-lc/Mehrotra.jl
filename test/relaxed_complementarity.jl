include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))


@testset "relaxed complementarity: random NCP (non negative cone)" begin
    for i = 0:10
        options = Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-10,
            complementarity_tolerance=(1/10)^i,
            compressed_search_direction=false,
            )

        solver = random_lcp(; num_primals=6, num_cone=12,
            cone_type=:non_negative_cone,
            options=options,
            seed=1,
            )

        Mehrotra.solve!(solver)
        @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
        @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
    end
end

@testset "relaxed complementarity: random NCP (second order cone)" begin
    for i = 0:10
        options = Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-10,
            complementarity_tolerance=(1/10)^i,
            compressed_search_direction=false,
            )

        solver = random_lcp(; num_primals=6, num_cone=12,
            cone_type=:second_order_cone,
            options=options,
            seed=1,
            )

        Mehrotra.solve!(solver)
        @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
        @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
    end
end


@testset "relaxed complementarity: contact ncp (linear_particle)" begin
    # dimensions and indices
    num_primals = 3
    num_cone = 6
    num_parameters = 14
    idx_nn = collect(1:6)
    idx_soc = [collect(1:0)]

    # parameters
    p2 = [1,1,1.0]
    v15 = [0,-1,1.0]
    u = [0.4, 0.8, 0.9]
    timestep = 0.01
    mass = 1.0
    gravity = -9.81
    friction_coefficient = 0.05
    side = 0.5

    parameters = [p2; v15; u; timestep; mass; gravity; friction_coefficient; side]

    # solve
    solver = Mehrotra.Solver(linear_particle_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            compressed_search_direction=false,
            )
        )
    # test relaxed complementarity
    for i = 0:10
        solver.options.complementarity_tolerance = (1/10)^i
        Mehrotra.solve!(solver)
        @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
        @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
    end
end


@testset "relaxed complementarity: contact ncp (nonlinear_particle)" begin
    # dimensions and indices
    num_primals = 3
    num_cone = 4
    num_parameters = 14
    idx_nn = collect(1:1)
    idx_soc = [collect(2:4)]

    # parameters
    p2 = [1,1,1.0]
    v15 = [0,-1,1.0]
    u = [0.4, 0.8, 0.9]
    timestep = 0.01
    mass = 1.0
    gravity = -9.81
    friction_coefficient = 0.5
    side = 0.5

    parameters = [p2; v15; u; timestep; mass; gravity; friction_coefficient; side]

    # solve
    solver = Mehrotra.Solver(non_linear_particle_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-10,
            complementarity_tolerance=1e-0,
            differentiate=false,
            compressed_search_direction=false,
            )
        )
    # test relaxed complementarity
    for i = 0:10
        solver.options.complementarity_tolerance = (1/10)^i
        Mehrotra.solve!(solver)
        @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
        @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
    end
end
