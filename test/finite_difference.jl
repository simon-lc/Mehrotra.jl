include("../examples/benchmark_problems/lcp_utils.jl")

@testset "finite difference: LCP" begin
    Random.seed!(0)

    num_primals = 10
    num_cone = 10
    num_parameters = num_primals^2 + 2num_primals*num_cone + num_primals + num_cone

    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    As = rand(num_primals, num_primals)
    A = As' * As
    B = rand(num_primals, num_cone)
    C = B'
    d = rand(num_primals)
    e = zeros(num_cone)
    parameters = [vec(A); vec(B); vec(C); d; e]

    dimensions = Dimensions(num_primals, num_cone, num_parameters)
    indices = Indices(num_primals, num_cone, num_parameters)

    # finite difference
    solver = Mehrotra.Solver(nothing, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        methods=Mehrotra.finite_difference_methods(lcp_residual, dimensions, indices),
        options=Mehrotra.Options(
            verbose=false,
            residual_tolerance=1e-6,
            complementarity_tolerance=1e-6,
            compressed_search_direction=false,
            )
        )

    Mehrotra.solve!(solver)
    finite_difference_iterations = solver.trace.iterations
    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance


    # symbolics
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
    symbolics_iterations = solver.trace.iterations
    equality_violation, cone_product_violation = Mehrotra.violation(solver)
    @test equality_violation <= solver.options.residual_tolerance
    @test cone_product_violation <= solver.options.residual_tolerance

    @test finite_difference_iterations == symbolics_iterations
end


Random.seed!(0)

num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + 2num_primals*num_cone + num_primals + num_cone

idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

As = rand(num_primals, num_primals)
A = As' * As
B = rand(num_primals, num_cone)
C = B'
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

dimensions = Dimensions(num_primals, num_cone, num_parameters)
indices = Indices(num_primals, num_cone, num_parameters)

# finite difference
solver = Mehrotra.Solver(nothing, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    methods=Mehrotra.finite_difference_methods(lcp_residual, dimensions, indices),
    options=Mehrotra.Options(
        verbose=false,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )
    )

Mehrotra.solve!(solver)
finite_difference_iterations = solver.trace.iterations
equality_violation, cone_product_violation = Mehrotra.violation(solver)
@test equality_violation <= solver.options.residual_tolerance
@test cone_product_violation <= solver.options.residual_tolerance


# symbolics
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
symbolics_iterations = solver.trace.iterations
equality_violation, cone_product_violation = Mehrotra.violation(solver)
@test equality_violation <= solver.options.residual_tolerance
@test cone_product_violation <= solver.options.residual_tolerance

@test finite_difference_iterations == symbolics_iterations
