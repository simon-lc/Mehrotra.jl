include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))
include("finite_difference_utils.jl")

################################################################################
# non negative cone
################################################################################
@testset "lcp: non negative cone" begin
    #########################################
    # problem setup
    #########################################
    solver = Mehrotra.random_lcp(;
        num_primals=2,
        num_cone=3,
        cone_type=:non_negative_cone,
        seed=1,
        options=Mehrotra.Options228(
            verbose=false,
            residual_tolerance=1e-8,
            complementarity_tolerance=1e-8,
            differentiate=true,
        ),
    )

    #########################################
    # residual jacobians
    #########################################
    test_residual_jacobian(solver; mode=:variables)
    @test test_residual_jacobian(solver; mode=:variables) < 1e-6

    test_residual_jacobian(solver; mode=:parameters)
    @test test_residual_jacobian(solver; mode=:parameters) < 1e-6

    #########################################
    # solution sensitivity
    #########################################
    test_solution_sensitivity(solver)
    @test test_solution_sensitivity(solver) < 1e-4
end

################################################################################
# second order cone
################################################################################
@testset "lcp: second order cone" begin
    #########################################
    # problem setup
    #########################################
    solver = Mehrotra.random_lcp(;
        num_primals=2,
        num_cone=3,
        cone_type=:second_order_cone,
        seed=1,
        options=Mehrotra.Options228(
            verbose=false,
            residual_tolerance=1e-8,
            complementarity_tolerance=1e-8,
            differentiate=true,
        ),
    )

    #########################################
    # residual jacobians
    #########################################
    test_residual_jacobian(solver; mode=:variables)
    @test test_residual_jacobian(solver; mode=:variables) < 1e-6

    test_residual_jacobian(solver; mode=:parameters)
    @test test_residual_jacobian(solver; mode=:parameters) < 1e-6

    #########################################
    # solution sensitivity
    #########################################
    test_solution_sensitivity(solver)
    @test test_solution_sensitivity(solver) < 1e-4
end




























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

################################################################################
# solve
################################################################################
solver = Solver(linear_particle_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(
        complementarity_tolerance=1e-10,
        residual_tolerance=1e-10,
        differentiate=true,
        verbose=true)
    )
