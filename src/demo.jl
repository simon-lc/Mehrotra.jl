using Mehrotra
using Random

include("../examples/benchmark_problems/lcp_utils.jl")

################################################################################
# coupled constraints
################################################################################
# dimensions
num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

# cone type
idx_nn = collect(1:num_cone-3)
idx_soc = [collect(num_cone-3+1:num_cone)]

# Jacobian
Random.seed!(0)
As = rand(num_primals, num_primals)
A = As' * As
b = rand(num_primals)
Cs = rand(num_cone, num_cone)
C = Cs * Cs'
d = rand(num_cone)
parameters = [vec(A); b; vec(C); d]

# solver
solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        compressed_search_direction=false,
        sparse_solver=false,
        differentiate=false,
        verbose=true,
        symmetric=false,
    ));


# solve
Mehrotra.solve!(solver)
solver.central_paths