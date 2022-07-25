using Mehrotra
using Random

include("../benchmark_problems/lcp_utils.jl")

################################################################################
# coupled constraints
################################################################################
# dimensions
num_primals = 1
num_cone = 1
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

# cone type
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

# Jacobian
Random.seed!(0)
As = rand(num_primals, num_primals)
A = As' * As
B = rand(num_primals, num_cone)
C = B'
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

# solver
solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    parameter_keywords=Dict(:x => 1:2),
    options=Options(
        compressed_search_direction=false,
        sparse_solver=false,
        differentiate=true,
        verbose=false,
        symmetric=false,
    ));

# J0 = solver.data.jacobian_parameters
# v0 = similar(J0, Int)
# fill!(J0, v0, :equality_jacobian_parameters)
# @benchmark $fill!($J0, $v0, :equality_jacobian_parameters)


# solve
Mehrotra.solve!(solver)
@benchmark $(Mehrotra.solve!)($solver)
solver.data.solution_sensitivity

solver.methods.equality_jacobian_keywords

@benchmark $differentiate!($solver; keywords=$([:all]))

solver.data

solver.methods

data = solver.data
problem = solver.problem
indices = solver.indices

Main.@code_warntype residual!(data, problem, indices;
        residual=false,
        jacobian_variables=false,
        jacobian_parameters=true,
        compressed=false,
        sparse_solver=false)

residual!(solver.data, solver.problem, solver.indices;
        residual=false,
        jacobian_variables=false,
        jacobian_parameters=false,
        compressed=false,
        sparse_solver=false)

@benchmark $residual!(
        $(data),
        $(problem),
        $(indices);
        residual=false,
        jacobian_variables=false,
        jacobian_parameters=true,
        compressed=false,
        sparse_solver=false)
