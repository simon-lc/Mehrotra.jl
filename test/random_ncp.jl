include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))

################################################################################
# non negative cone
################################################################################
@testset "random ncp: non negative cone" begin
    Random.seed!(0)

    num_primals = 10
    num_cone = 10
    num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    As = rand(num_primals, num_primals)
    A = As' * As
    B = rand(num_primals, num_cone)
    C = B'
    d = rand(num_primals)
    e = zeros(num_cone)
    parameters = [vec(A); vec(B); vec(C); d; e]


    solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            compressed_search_direction=false,
            )
        )

    Mehrotra.solve!(solver)
    equality_violation, cone_product_violation =
        Mehrotra.violation(solver.problem, solver.central_paths.tolerance_central_path)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance

    # num_allocs = @ballocated $(Mehrotra.solve!)($solver)
    # @test (num_allocs == 0) broken=!(num_allocs == 0)
end


################################################################################
# non negative cone
################################################################################
@testset "random ncp: second order cone" begin
    Random.seed!(0)

    num_primals = 4
    num_cone = 15

    idx_nn = collect(1:0)
    idx_soc = [collect(1:3), collect(4:6), collect(7:9), collect(10:12), collect(13:15)]

    As = rand(num_primals, num_primals)
    A = As' * As
    B = rand(num_primals, num_cone)
    C = B'
    d = rand(num_primals)
    e = zeros(num_cone)
    parameters = [vec(A); vec(B); vec(C); d; e]

    solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            compressed_search_direction=false,
            )
        )

    Mehrotra.solve!(solver)
    equality_violation, cone_product_violation =
        Mehrotra.violation(solver.problem, solver.central_paths.tolerance_central_path)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance

    # solver.options.verbose = false
    # num_allocs = @ballocated $(Mehrotra.solve!)($solver)
    # @test (num_allocs == 0) broken=!(num_allocs == 0)
end

include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))


Random.seed!(0)

num_primals = 1
num_cone = 2#15

idx_nn = collect(1:0)
idx_soc = [collect(1:2)]#, collect(4:6)]#, collect(7:9)]#, collect(10:12), collect(13:15)]

As = ones(num_primals, num_primals)
A = As' * As
B = ones(num_primals, num_cone)
C = B'
d = ones(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Mehrotra.Options(
        verbose=true,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-2,
        compressed_search_direction=false,
        complementarity_decoupling=false,
        differentiate=false,
        )
    )

solve!(solver)

solver.data

solver.problem.cone_product
solver.data.residual.all
solver.step_sizes

solver.central_paths
solver.data.step
solver.data.residual.all

z0 = solver.solution.duals
s0 = solver.solution.slacks
zs = cone_product(z0, s0, idx_nn, idx_soc)
zz = cone_product(z0, z0, idx_nn, idx_soc)
ss = cone_product(s0, s0, idx_nn, idx_soc)
for ind in idx_soc
    @show ss[ind[1]] - norm(ss[ind][2:3])
end
for ind in idx_soc
    @show zz[ind[1]] - norm(zz[ind][2:3])
end


solver.problem.cone_product

solver.central_paths.tolerance_central_path
solver.problem.cone_product - solver.central_paths.tolerance_central_path
Mehrotra.solve!(solver)
equality_violation, cone_product_violation =
    Mehrotra.violation(solver.problem, solver.central_paths.tolerance_central_path)
@test equality_violation <= solver.options.residual_tolerance
@test cone_product_violation <= solver.options.residual_tolerance


cone_violation(solver)















Random.seed!(0)

num_primals0 = 1
num_cone0 = 3

idx_nn0 = collect(1:0)
# idx_soc0 = [collect(1:3), collect(4:6), collect(7:9), collect(10:12), collect(13:15)]
idx_soc0 = [collect(1:3)]

As = rand(num_primals0, num_primals0)
A0 = As' * As
B0 = rand(num_primals0, num_cone0)
C0 = B0'
d0 = rand(num_primals0)
e0 = zeros(num_cone0)
parameters0 = [vec(A0); vec(B0); vec(C0); d0; e0]

solver0 = Mehrotra.Solver(lcp_residual, num_primals0, num_cone0,
    parameters=parameters0,
    nonnegative_indices=idx_nn0,
    second_order_indices=idx_soc0,
    options=Mehrotra.Options(
        # verbose=false,
        verbose=true,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )
    )
initialize_solver!(solver0)
solver0.central_paths
solver0.central_paths.target_central_path
correction!(solver0.data, solver0.methods, solver0.solution, solver0.central_paths.target_central_path)
solver0.central_paths.target_central_path


solve!(solver0)
solver0.central_paths.target_central_path
equality_violation, cone_product_violation =
    Mehrotra.violation(solver.problem, solver.central_paths.tolerance_central_path)
@test equality_violation <= solver.options.residual_tolerance
@test cone_product_violation <= solver.options.residual_tolerance

# solver.options.verbose = false
# num_allocs = @ballocated $(Mehrotra.solve!)($solver)
# @test (num_allocs == 0) broken=!(num_allocs == 0)

















Random.seed!(0)

num_primals = 4
num_cone = 15
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

idx_nn = collect(1:0)
idx_soc = [collect(1:3), collect(4:6), collect(7:9), collect(10:12), collect(13:15)]

idx_nn = collect(1:15)
idx_soc = [collect(1:0)]

As = rand(num_primals, num_primals)
A = As' * As
B = rand(num_primals, num_cone)
C = B'
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Mehrotra.Options(
        # verbose=false,
        verbose=true,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=true,
        )
    )

Mehrotra.solve!(solver)
equality_violation, cone_product_violation =
    Mehrotra.violation(solver.problem, solver.central_paths.tolerance_central_path)
@test equality_violation <= solver.options.residual_tolerance
@test cone_product_violation <= solver.options.residual_tolerance
















Random.seed!(0)

num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

As = rand(num_primals, num_primals)
A = As' * As
B = rand(num_primals, num_cone)
C = B'
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]


solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Mehrotra.Options(
        verbose=false,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )
    )

Mehrotra.solve!(solver)
equality_violation, cone_product_violation =
    Mehrotra.violation(solver.problem, solver.central_paths.tolerance_central_path)
@test equality_violation <= solver.options.residual_tolerance
@test cone_product_violation <= solver.options.residual_tolerance

num_allocs = @ballocated $(Mehrotra.solve!)($solver)
@test (num_allocs == 0) broken=!(num_allocs == 0)
