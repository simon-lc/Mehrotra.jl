using Mehrotra
using Random

include("../polytope/contact_model/lp_2d.jl");


# parameters
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2)
bp = 0.5*[
    +1,
    +1,
    +1,
     1,
     # 2,
    ]
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.00ones(4,2)
bc = 2.0*[
     1,
     1,
     1,
     1,
    ]

solver = lp_contact_solver(Ap, bp, Ac, bc; d=2,
    options=Options(
        verbose=false,
        complementarity_tolerance=3e-3,
        residual_tolerance=1e-6,
        differentiate=true,
        compressed_search_direction=false,
        sparse_solver=true,
        ));

d = 2
xp = [40, 40, 0.0]
xc = [00, 00, 0.0]
np = length(bp)
nc = length(bc)

solver.parameters[1:2d+2] .= [xp; xc]
solve!(solver)
solver.solution.primals
zp = solver.solution.duals[1:np]
zc = solver.solution.duals[np .+ (1:nc)]

x_2d_rotation(xp[3:3]) * Ap' * zp
x_2d_rotation(xc[3:3]) * Ac' * zc


solution_sensitivity = solver.data.solution_sensitivity
N = solution_sensitivity[d .+ (1:1), 1:2d+2]

using Plots
plot(Gray.(1e3abs.(solution_sensitivity)))


# Main.@profview [solve!(solver) for i=1:10000]
# @benchmark $solve!($solver)


solver.problem.equality_jacobian_parameters_sparse
solver.problem.equality_jacobian_variables_sparse[1:7,1:7]
solver.problem.cone_product_jacobian_duals_sparse
solver.problem.cone_product_jacobian_slacks_sparse

solver.data.jacobian_variables
solver.data.jacobian_variables_sparse.matrix

solver.options.sparse_solver
solver.linear_solver

solver.data.jacobian_variables
solver.data.jacobian_variables_sparse.matrix
solver.data.jacobian_variables_dense
