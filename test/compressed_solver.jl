@testset "compressed solver" begin
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
    solveru = Mehrotra.Solver(linear_particle_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-10,
            complementarity_tolerance=1e-6,
            compressed_search_direction=false,
            )
        )

    solverc = Mehrotra.Solver(linear_particle_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-10,
            complementarity_tolerance=1e-6,
            compressed_search_direction=true,
            )
        )
    Mehrotra.solve!(solveru)
    Mehrotra.solve!(solverc)
    @test norm(solveru.solution.all - solverc.solution.all) < 1e-8
    norm(solveru.solution.all - solverc.solution.all)

    dimensions = solverc.dimensions
    variables = rand(dimensions.variables)

    Mehrotra.initialize_variables!(solverc, variables)
    Mehrotra.initialize_variables!(solveru, variables)

    Mehrotra.differentiate!(solverc)
    Mehrotra.differentiate!(solveru)

    S0 = solverc.data.solution_sensitivity
    S1 = solveru.data.solution_sensitivity
    @test norm(S0 - S1) < 1e-10

    Mehrotra.search_direction!(solveru)
    Δ0 = solveru.data.step.all
    Mehrotra.search_direction!(solverc)
    Δ1 = solverc.data.step.all
    @test norm(Δ0 - Δ1) < 1e-10
end
