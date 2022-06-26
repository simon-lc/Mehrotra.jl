include(joinpath(module_dir(), "examples/benchmark_problems/particle_utils.jl"))
include(joinpath(module_dir(), "examples/benchmark_problems/block_2d_utils.jl"))

################################################################################
# linear particle
################################################################################
@testset "contact ncp: linear particle" begin
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
        options=Mehrotra.Options228(
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            )
        )

    Mehrotra.solve!(solver)
    @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
    @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
end


################################################################################
# non linear particle
################################################################################
@testset "contact ncp: non linear particle" begin
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
        options=Mehrotra.Options228(
            max_iterations=30,
            verbose=true,
            complementarity_tolerance=1e-4,
            residual_tolerance=1e-4,
            differentiate=false,
            )
        )

    Mehrotra.solve!(solver)
    @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
    @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
end
