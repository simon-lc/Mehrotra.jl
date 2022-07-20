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
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            compressed_search_direction=false,
            )
        )
    Mehrotra.solve!(solveru)

    solverc = Mehrotra.Solver(linear_particle_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            compressed_search_direction=true,
            )
        )
    Mehrotra.solve!(solverc)
    dimensions = solverc.dimensions
    variables = rand(dimensions.variables)

    Mehrotra.initialize!(solverc, variables)
    Mehrotra.initialize!(solveru, variables)

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


#
#
# # dimensions and indices
# num_primals = 3
# num_cone = 4
# num_parameters = 14
# idx_nn = collect(1:1)
# idx_soc = [collect(2:4)]
#
# # parameters
# p2 = [1,1,1.0]
# v15 = [0,-1,1.0]
# u = [0.4, 0.8, 0.9]
# timestep = 0.01
# mass = 1.0
# gravity = -9.81
# friction_coefficient = 0.5
# side = 0.5
#
# parameters = [p2; v15; u; timestep; mass; gravity; friction_coefficient; side]
#
# # solve
# solveru = Mehrotra.Solver(non_linear_particle_residual, num_primals, num_cone,
#     parameters=parameters,
#     nonnegative_indices=idx_nn,
#     second_order_indices=idx_soc,
#     options=Mehrotra.Options(
#         verbose=true,
#         residual_tolerance=1e-6,
#         complementarity_tolerance=1e-6,
#         differentiate=false,
#         compressed_search_direction=false,
#         )
#     )
# Mehrotra.solve!(solveru)
#
# solverc = Mehrotra.Solver(non_linear_particle_residual, num_primals, num_cone,
#     parameters=parameters,
#     nonnegative_indices=idx_nn,
#     second_order_indices=idx_soc,
#     options=Mehrotra.Options(
#         verbose=true,
#         max_iterations=8,
#         residual_tolerance=1e-6,
#         complementarity_tolerance=1e-6,
#         differentiate=false,
#         compressed_search_direction=true,
#         )
#     )
# Mehrotra.solve!(solverc)
# dimensions = solverc.dimensions
# variables = rand(dimensions.variables)
# variables = [ones(3); [1,1,0.0,0.0]; [1,1,0.0,0.0]]
# Mehrotra.initialize!(solverc, variables)
# Mehrotra.initialize!(solveru, variables)
#
# Mehrotra.differentiate!(solverc)
# Mehrotra.differentiate!(solveru)
#
# S0 = solverc.data.solution_sensitivity
# S1 = solveru.data.solution_sensitivity
#
# solverc.data.residual_compressed
# solverc.data.jacobian_variables_compressed_sparse
# solverc.data.jacobian_variables
# inv(Matrix(solverc.data.jacobian_variables_compressed_sparse))
# @test norm(S0 - S1) < 1e-10
# norm(S0 - S1, Inf)
#
# Mehrotra.search_direction!(solveru)
# Δ0 = solveru.data.step.all
# Mehrotra.search_direction!(solverc)
# Δ1 = solverc.data.step.all
# @test norm(Δ0 - Δ1, Inf) < 1e-10
# norm(Δ0 - Δ1, Inf)
#
#
# solverc.data.jacobian_variables_compressed_sparse
# Δ0
# Δ1
# Δ0 - Δ1
#
#
# S0
# S1
# S0 - S1
#
#
#
#
# norm(solverc.data.residual.all - solveru.data.residual.all)
# norm(solverc.data.residual_compressed.all - solveru.data.residual.all)
#
# solverc.data.residual_compressed.all - solveru.data.residual.all
#
#
# Zi = solverc.data.cone_product_jacobian_inverse_slack
# ZiS = solverc.data.cone_product_jacobian_ratio
# S = solverc.data.cone_product_jacobian_duals
# inv(S)
# Zi * S
#
# S = solverc.problem.cone_product_jacobian_duals
# Z = solverc.problem.cone_product_jacobian_slacks
# Si = solverc.problem.cone_product_jacobian_inverse_dual
#
# Zi = solverc.problem.cone_product_jacobian_inverse_slack
#
# norm(Zi * Z - I, Inf)
# norm(Si * S - I, Inf)
#
# solverc.problem
#
# z = solverc.solution.duals
# s = solverc.solution.slacks
# function coneprod(z, s)
#     p = zeros(4)
#     solverc.cone_methods.product(p, z, s)
#     return p
# end
#
# norm(FiniteDiff.finite_difference_jacobian(z -> coneprod(z,s), z) - S, Inf)
# norm(FiniteDiff.finite_difference_jacobian(s -> coneprod(z,s), s) - Z, Inf)
#
# solverc.problem
