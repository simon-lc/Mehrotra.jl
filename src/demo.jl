using Mehrotra
using Random

include("../examples/benchmark_problems/lcp_utils.jl")

################################################################################
# coupled constraints
################################################################################
# dimensions
num_primals = 1
num_cone = 1
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

# cone type
# idx_nn = collect(1:num_cone-3)
# idx_soc = [collect(num_cone-3+1:num_cone)]
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

# Jacobian
Random.seed!(0)
As = rand(num_primals, num_primals)
A = As' * As
B = rand(num_primals, num_cone)
C = rand(num_cone, num_primals)
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

# solver
solver0 = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        compressed_search_direction=false,
        sparse_solver=true,
        differentiate=false,
        verbose=true,
        symmetric=false,
        max_iterations=1000,
    ));

solve!(solver0)

solver1 = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        compressed_search_direction=true,
        sparse_solver=false,
        differentiate=false,
        verbose=true,
        symmetric=false,
        max_iterations=100,
    ));

solve!(solver1)
solver1.solution.all


solver1.data.step.all
solver1.data.residual.all
solver1.data.residual_compressed.all
solver1.data.jacobian_variables_compressed_dense
solver1.problem.equality_constraint
solver1.problem.equality_constraint_compressed


residual!(solver1.data, solver1.problem, solver1.indices, compressed=true)
correction!(solver1.data, solver1.methods, solver1.solution, solver1.central_paths.central_path; compressed=true)

solver1.data.residual_compressed.all

initialize_solver!(solver0)
initialize_solver!(solver1)
# solve
# Mehrotra.solve!(solver0)
# Mehrotra.solve!(solver1)
evaluate!(solver1.problem,
        solver1.methods,
        solver1.cone_methods,
        solver1.solution,
        solver1.parameters;
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=false,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        sparse_solver=true,
        compressed=solver1.options.compressed_search_direction,
        )

evaluate!(solver0.problem,
        solver0.methods,
        solver0.cone_methods,
        solver0.solution,
        solver0.parameters;
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=false,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        sparse_solver=true,
        compressed=solver0.options.compressed_search_direction,
        )

solver0.problem.equality_constraint
solver0.problem.equality_jacobian_variables




solver1.problem.equality_constraint
solver1.problem.equality_constraint_compressed
solver1.problem.equality_jacobian_variables_compressed

















solver0.problem.cone_product
solver1.problem.cone_product

solver1.data







solver.data.step.all
