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
    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance

    num_allocs = @ballocated $(Mehrotra.solve!)($solver)
    @test (num_allocs == 0) broken=!(num_allocs == 0)
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
    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance

    solver.options.verbose = false
    num_allocs = @ballocated $(Mehrotra.solve!)($solver)
    @test (num_allocs == 0) broken=!(num_allocs == 0)
end



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
        complementarity_decoupling=true,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )
    )

Mehrotra.solve!(solver)
equality_violation, cone_product_violation = Mehrotra.violation(solver)
@test equality_violation <= solver.options.residual_tolerance
@test cone_product_violation <= solver.options.residual_tolerance

num_allocs = @ballocated $(Mehrotra.solve!)($solver)
@test (num_allocs == 0) broken=!(num_allocs == 0)
