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
B = rand(num_primals, num_cone)
C = rand(num_cone, num_primals)
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

# solver
solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(
        compressed_search_direction=false,
        sparse_solver=true,
        differentiate=true,
        verbose=true,
        symmetric=false,
    ));

solver.linear_solver
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
B = rand(num_primals, num_cone)
C = rand(num_cone, num_primals)
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

# solver
solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(complementarity_decoupling=true),
    )

# solve
Mehrotra.solve!(solver)



n = 10
a = zeros(2n)
b = zeros(2n)
av = @views a[Vector(1:n)]
bv = @views b[Vector(1:n)]

num_primals = 10
num_cone = 10
num_parameters = 10

dim = Dimensions(num_primals, num_cone, num_parameters)
idx = Indices(num_primals, num_cone, num_parameters)
point = Point(dim, idx)
A = sprand(n,n,1.0)
point.primals .= A \ point.duals
typeof(av)
