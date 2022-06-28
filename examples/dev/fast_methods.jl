using Mehrotra
using BenchmarkTools

include(joinpath(module_dir(), "examples/benchmark_problems/lcp_utils.jl"))

num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

idx_nn = collect(1:5)
idx_soc = [collect(6:8), collect(9:10)]

# idx_nn = collect(1:10)
# idx_soc = [collect(1:0)]
idx_soc = []

As = rand(num_primals, num_primals)
A = As' * As
b = rand(num_primals)
Cs = rand(num_cone, num_cone)
C = Cs * Cs'
d = rand(num_cone)
parameters = [vec(A); b; vec(C); d]

solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(
        verbose=false,
        differentiate=true,
        compressed_search_direction=true,
        )
    )


solve!(solver)
Main.@code_warntype solve!(solver)
@benchmark $solve!($solver)

















linear_solver = solver.linear_solver
dimensions = solver.dimensions
data = solver.data
residual = data.residual
stepp = data.step

compressed_search_direction!(linear_solver, dimensions, data, residual, stepp)
Main.@code_warntype compressed_search_direction!(linear_solver, dimensions, data, residual, stepp)
@benchmark $compressed_search_direction!($linear_solver, $dimensions, $data, $residual, $stepp)

search_direction!(solver, compressed=true)
Main.@code_warntype search_direction!(solver, compressed=true)
@benchmark $search_direction!($solver, compressed=true)



data = solver.data
problem = solver.problem
indices = solver.indices
solution = solver.solution
parameters = solver.parameters
zero_central_path = solver.central_paths.zero_central_path

residual!(data, problem, indices, solution, parameters,
    zero_central_path, compressed=true)

Main.@code_warntype residual!(data, problem, indices, solution, parameters,
    zero_central_path, compressed=true)

@benchmark residual!($data, $problem, $indices, $solution, $parameters,
    $zero_central_path, compressed=true)


z = rand(num_cone)
s = rand(num_cone)
S = cone_product_jacobian(z, s, idx_nn, idx_soc)
Zi = cone_product_jacobian_inverse(s, z, idx_nn, idx_soc)

using Plots
Zi * S
plot(Gray.(100*abs.(Zi)))
plot(Gray.(100*abs.(S)))
plot(Gray.(100*abs.(Zi*S)))

solver.data.cone_product_jacobian_dual
solver.data.cone_product_jacobian_ratio
solver.data.cone_product_jacobian_ratio

solver.data.compressed_jacobian_variables

typeof(solver.solution.equality)

Main.@profiler begin
    for i = 1:1000
        solve!(solver)
    end
end

linear_solver = solver.linear_solver
solution = solver.solution
data = solver.data
residual = solver.data.residual
@benchmark $linear_solve!($linear_solver, $solution.equality,
    $data.dense_compressed_jacobian_variables, $residual.equality)

Main.@code_warntype linear_solve!(linear_solver, solution.equality,
    data.dense_compressed_jacobian_variables, residual.equality)

search_direction!(solver, compressed=true)
Main.@code_warntype search_direction!(solver, compressed=true)
@benchmark $search_direction!($solver, compressed=true)


################################################################################
# residual
################################################################################
data = solver.data
problem = solver.problem
methods = solver.methods
cone_methods = solver.cone_methods
solution = solver.solution
parameters = solver.parameters
indices = solver.indices
κ = [1e-4]

residual!(data, problem, indices, solution, parameters, κ)
Main.@code_warntype residual!(data, problem, indices, solution, parameters, κ)

@benchmark $residual!($data, $problem, $indices, $solution, $parameters, $κ)


A = rand(3,3)
B = rand(3,3)
C = A * B
B .*= A
B - C

################################################################################
# Evaluate
################################################################################
problem = solver.problem
methods = solver.methods
cone_methods = solver.cone_methods
solution = solver.solution
parameters = solver.parameters

# evaluate
evaluate!(problem, methods, cone_methods, solution, parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    equality_jacobian_parameters=true,
    cone_constraint=true,
    cone_jacobian_variables=true,
)
Main.@code_warntype evaluate!(problem, methods, cone_methods, solution, parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    equality_jacobian_parameters=true,
    cone_constraint=true,
    cone_jacobian_variables=true,
)
@benchmark evaluate!($problem, $methods, $cone_methods, $solution, $parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    equality_jacobian_parameters=true,
    cone_constraint=true,
    cone_jacobian_variables=true,
)
