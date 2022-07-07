using Mehrotra
using Random

include("lcp_utils.jl")

################################################################################
# coupled constraints
################################################################################
# dimensions
num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

# cone type
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

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
    )

# solve
Mehrotra.solve!(solver)



################################################################################
# decoupled constraints
################################################################################
# dimensions
num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

# cone type
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

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
    options=Options228(complementarity_decoupling=true),
    )

# solve
Mehrotra.solve!(solver)
