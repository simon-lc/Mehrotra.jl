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
C = B'
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

# solver
solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(
        compressed_search_direction=true,
        sparse_solver=false,
        differentiate=true,
        verbose=false,
        symmetric=false,
    ));

solver
solver.linear_solver
# solve
Mehrotra.solve!(solver)

@benchmark $solve!($solver)


solver.options.sparse_solver
solver.options.compressed_search_direction

search_direction!(solver)
Main.@code_warntype search_direction!(solver)
@benchmark $search_direction!($solver)





linear_solver = solver.linear_solver
data = solver.data
problem = solver.problem
indices = solver.indices
methods0 = solver.methods
affine_step_size = solver.step_sizes.affine_step_size
step0 = solver.data.step
solution= solver.solution
tolerance_central_path = solver.central_paths.tolerance_central_path
step_correction = solver.data.step_correction
central_path = solver.central_paths.tolerance_central_path

compressed_search_direction!(
    linear_solver,
    data,
    step0,
    methods0,
    solution,
    sparse_solver=false)

Main.@code_warntype compressed_search_direction!(
    linear_solver,
    data,
    step0,
    methods0,
    solution,
    sparse_solver=false)

@benchmark $compressed_search_direction!(
    $linear_solver,
    $data,
    $step0,
    $methods0,
    $solution,
    sparse_solver=false)






residual!(data, problem, indices,
    compressed=true,
    sparse_solver=false)

Main.@code_warntype correction!(methods, data, affine_step_size, step0, step_correction, solution, tolerance_central_path;
    compressed=true, complementarity_correction=0.0)

correction!(methods, data, affine_step_size, step0, step_correction, solution, tolerance_central_path;
    compressed=true, complementarity_correction=0.0)

@benchmark $correction!($methods, $data, $affine_step_size, $step0, $step_correction, $solution, $tolerance_central_path;
    compressed=true, complementarity_correction=0.0)

@benchmark $(methods.correction_compressed)(
    $(data.residual_compressed.all),
    $(data.residual_compressed.all),
    $(step_correction.duals),
    $(step_correction.slacks),
    $(solution.all),
    $(central_path),
    )

function corr0(methods::ProblemMethods, data::SolverData, Δz, Δs, solution::Point, κ; compressed::Bool=false)
    methods.correction(data.residual.all, data.residual.all, Δz, Δs, κ)
    if compressed
        methods.correction_compressed(data.residual_compressed.all, data.residual_compressed.all, Δz, Δs, solution.all, κ)
    end
    return nothing
end



r0 = rand(30)
Δz0 = solver.data.step_correction.duals
Δs0 = solver.data.step_correction.slacks
x0 = rand(30)
κ0 = rand(10)
residual0 = solver.data.residual
solution0 = solver.solution
data0 = solver.data
@benchmark $corr0($methods, $data0, $Δz0, $Δs0, $solution0, $κ0)
# @benchmark $corr0($methods, $r0, $Δz0, $Δs0, $x0, $κ0)

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
C = B'
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
