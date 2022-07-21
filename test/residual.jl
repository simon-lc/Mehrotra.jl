include("../examples/benchmark_problems/lcp_utils.jl")
include("finite_difference_utils.jl")

@testset "residual" begin
    Random.seed!(0)

    num_primals = 10
    num_cone = 10
    num_variables = num_primals + 2num_cone
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

    dimensions = Dimensions(num_primals, num_cone, num_parameters)
    indices = Indices(num_primals, num_cone, num_parameters)

    #non compressed
    for sparse_solver in (true, false)
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
                sparse_solver=sparse_solver,
                )
            );

        solver.solution.all .= rand(num_variables)

        evaluate!(solver.problem,
            solver.methods,
            solver.cone_methods,
            solver.solution,
            solver.parameters;
            equality_constraint=true,
            equality_jacobian_variables=false,
            equality_jacobian_parameters=false,
            cone_constraint=true,
            cone_jacobian=false,
            cone_jacobian_inverse=false,
            sparse_solver=false,
            compressed=false,
            )

        r0 = extended_residual(lcp_residual, indices, solver.solution.all, solver.parameters)
        r1 = [solver.problem.equality_constraint; solver.problem.cone_product]
        @test norm(r0 - r1, Inf) < 1e-10

    end

    # compressed
    for sparse_solver in (true, false)
        # symbolics
        solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
            parameters=parameters,
            nonnegative_indices=idx_nn,
            second_order_indices=idx_soc,
            options=Mehrotra.Options(
                verbose=false,
                residual_tolerance=1e-6,
                complementarity_tolerance=1e-6,
                compressed_search_direction=true,
                sparse_solver=sparse_solver,
                )
            );

        solver.solution.all .= rand(num_variables)

        evaluate!(solver.problem,
            solver.methods,
            solver.cone_methods,
            solver.solution,
            solver.parameters;
            equality_constraint=true,
            equality_jacobian_variables=false,
            equality_jacobian_parameters=false,
            cone_constraint=true,
            cone_jacobian=false,
            cone_jacobian_inverse=false,
            sparse_solver=false,
            compressed=true,
            )

        r0 = extended_residual(lcp_residual, indices, solver.solution.all, solver.parameters)
        e0 = r0[indices.equality]
        c0 = r0[indices.cone_product]

        e1 = solver.problem.equality_constraint_compressed
        c1 = solver.problem.cone_product
        @test norm((e0-e1)[indices.optimality], Inf) < 1e-10
        @test norm(c0 - c1, Inf) < 1e-10
    end
end
