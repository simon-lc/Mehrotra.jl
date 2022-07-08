include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))

################################################################################
# non negative cone
################################################################################
@testset "random ncp: non negative cone" begin
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

    solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
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

    Mehrotra.solve!(solver)
    @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
    @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance

    num_allocs = @ballocated $(Mehrotra.solve!)($solver)
    @test (num_allocs == 0) broken=!(num_allocs == 0)
end


################################################################################
# non negative cone
################################################################################
@testset "random ncp: second order cone" begin
    Random.seed!(0)

    num_primals = 4
    num_cone = 15
    num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

    idx_nn = collect(1:0)
    idx_soc = [collect(1:3), collect(4:6), collect(7:9), collect(10:12), collect(13:15)]

    As = rand(num_primals, num_primals)
    A = As' * As
    b = rand(num_primals)
    Cs = rand(num_cone, num_cone)
    C = Cs * Cs'
    d = rand(num_cone)
    parameters = [vec(A); b; vec(C); d]

    solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
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

    Mehrotra.solve!(solver)
    @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
    @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance

    solver.options.verbose = false
    num_allocs = @ballocated $(Mehrotra.solve!)($solver)
    @test (num_allocs == 0) broken=!(num_allocs == 0)
end


#
# Random.seed!(0)
#
# num_primals = 10
# num_cone = 10
# num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone
#
# idx_nn = collect(1:num_cone)
# idx_soc = [collect(1:0)]
#
# As = rand(num_primals, num_primals)
# A = As' * As
# b = rand(num_primals)
# Cs = rand(num_cone, num_cone)
# C = Cs * Cs'
# d = rand(num_cone)
# parameters = [vec(A); b; vec(C); d]
#
# solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
#     parameters=parameters,
#     nonnegative_indices=idx_nn,
#     second_order_indices=idx_soc,
#     options=Mehrotra.Options(
#         verbose=false,
#         residual_tolerance=1e-6,
#         complementarity_tolerance=1e-6,
#         compressed_search_direction=true,
#         differentiate=false,
#         )
#     )
#
# Mehrotra.solve!(solver)
# @test norm(solver.data.residual.equality, Inf) <= solver.options.residual_tolerance
# @test Mehrotra.cone_violation(solver) <= solver.options.residual_tolerance
#
# @benchmark $solve!($solver)
# Main.@profiler [solve!(solver) for i=1:1000]
#
# search_direction!(solver)
# @benchmark $search_direction!($solver)
#
# dimensions = solver.dimensions
# linear_solver = solver.linear_solver
# data = solver.data
# step0 = data.step
# compressed = solver.options.compressed_search_direction
#
#
# compressed_search_direction!(linear_solver, dimensions, data, step0)
# Main.@code_warntype compressed_search_direction!(linear_solver, dimensions, data, step0)
# @benchmark $compressed_search_direction!($linear_solver, $dimensions, $data, $step0)
#
# solver.data
#
# data = solver.data
# problem = solver.problem
# indices = solver.indices
# solution = solver.solution
# parameters = solver.parameters
# central_paths = solver.central_paths
#
#
#
#
#
# residual!(data, problem, indices, solution, parameters,
#     central_paths.target_central_path; compressed=true)
# Main.@code_warntype residual!(data, problem, indices, solution, parameters,
#     central_paths.target_central_path; compressed=true)
# @benchmark $residual!($data, $problem, $indices, $solution, $parameters,
#     $central_paths.target_central_path; compressed=true)
