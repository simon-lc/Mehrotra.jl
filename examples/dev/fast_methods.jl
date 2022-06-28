using Mehrotra
using BenchmarkTools

include(joinpath(module_dir(), "examples/benchmark_problems/lcp_utils.jl"))

num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

# idx_nn = collect(1:5)
# idx_soc = [collect(6:8), collect(9:10)]

idx_nn = collect(1:10)
idx_soc = [collect(1:0)]
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
        differentiate=false,
        )
    )

solve!(solver)
Main.@code_warntype solve!(solver)
@benchmark $solve!($solver)


solver.data.jacobian_variables



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
