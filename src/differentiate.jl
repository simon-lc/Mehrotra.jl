function differentiate!(solver)
    data = solver.data
    problem = solver.problem
    methods = solver.methods
    cone_methods = solver.cone_methods
    indices = solver.indices
    dimensions = solver.dimensions
    solution = solver.solution
    parameters = solver.parameters
    options = solver.options
    κ = solver.central_paths
    compressed = options.compressed_search_direction
    sparse_solver = options.sparse_solver

    # Here we only need to update
    # equality_jacobian_variables
    # equality_jacobian_parameters
    # cone_jacobian_variables
    # evaluate derivatives wrt to parameters
    # # TODO: check if we can use current residual Jacobian w/o recomputing
    Mehrotra.evaluate!(problem,
        methods,
        cone_methods,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
    )

    residual!(data, problem, indices, κ.tolerance_central_path,
        compressed=compressed, sparse_solver=sparse_solver)

    # # residual Jacobian wrt variables
    # residual_jacobian_variables!(solver.data, solver.problem, solver.indices,
    #     solver.central_path, solver.penalty, solver.dual,
    #     solver.primal_regularization, solver.dual_regularization,
    #     constraint_tensor=solver.options.constraint_tensor)
    # residual_jacobian_variables_symmetric!(solver.data.jacobian_variables_symmetric, solver.data.jacobian_variables, solver.indices,
    #     solver.problem.second_order_jacobians, solver.problem.second_order_jacobians_inverse)
    # factorize!(solver.linear_solver, solver.data.jacobian_variables_symmetric;
    #     update=solver.options.update_factorization)

    # # residual Jacobian wrt parameters
    # residual_jacobian_parameters!(solver.data, solver.problem, solver.indices)

    # compute solution sensitivities
    fill!(solver.data.solution_sensitivity, 0.0)
    # data.solution_sensitivity .= - data.jacobian_variables \ data.jacobian_parameters #TODO
    if compressed
        Zi = data.cone_product_jacobian_inverse_slack
        S = data.cone_product_jacobian_duals

        # primal dual step
        data.jacobian_variables_dense_compressed .= data.jacobian_variables_sparse_compressed
        linear_solve!(solver.linear_solver,
            # data.solution_sensitivity[indices.equality, :],
            view(data.solution_sensitivity, 1:dimensions.equality, :),
            data.jacobian_variables_dense_compressed,
            # data.jacobian_parameters[indices.equality, :],
            view(data.jacobian_parameters, 1:dimensions.equality, :),
            fact=true)
        data.solution_sensitivity .*= -1.0

        # for i = 1:dimensions.parameters
        #     data.solution_sensitivity[indices.slacks, i] .=
        #         -Zi * (data.jacobian_parameters[indices.cone_product, i] + S * data.solution_sensitivity[indices.duals, i]) # -Z⁻¹ (cone_product + S * Δz)
        #     end

        # slack step
        data.solution_sensitivity[indices.slacks, :] .=
            -Zi * (data.jacobian_parameters[indices.cone_product, :] + S * data.solution_sensitivity[indices.duals, :]) # -Z⁻¹ (cone_product + S * Δz)
            # -Zi * (data.residual.cone_product .+ (S * data.solution_sensitivity[indices.duals, :])) # -Z⁻¹ (cone_product + S * Δz)
            # -Zi * (hcat([data.residual.cone_product for i=1:dimensions.parameters]...) + (S * data.solution_sensitivity[indices.duals, :])) # -Z⁻¹ (cone_product + S * Δz)
    else
        if options.sparse_solver
            linear_solve!(solver.linear_solver, data.solution_sensitivity,
                data.jacobian_variables_sparse.matrix, data.jacobian_parameters, fact=true)
            data.solution_sensitivity .*= -1.0
        else
            data.jacobian_variables_dense .= data.jacobian_variables
            linear_solve!(solver.linear_solver, data.solution_sensitivity,
                data.jacobian_variables_dense, data.jacobian_parameters, fact=true)
            data.solution_sensitivity .*= -1.0
        end
    end
    # #TODO parallelize, make more efficient
    # for i in solver.indices.parameters
    #     for k = 1:solver.dimensions.total
    #         solver.data.jacobian_parameters_vectors[i].all[k] = solver.data.jacobian_parameters[k, i]
    #     end
    #
    #     if solver.options.linear_solver == :QDLDL
    #         search_direction_symmetric!(
    #             solver.data.solution_sensitivity_vectors[i],
    #             solver.data.jacobian_parameters_vectors[i],
    #             solver.data.jacobian_variables,
    #             solver.data.step_symmetric,
    #             solver.data.residual_symmetric,
    #             solver.data.jacobian_variables_symmetric,
    #             solver.indices,
    #             solver.data.solution_sensitivity_vectors_second_order[i],
    #             solver.data.jacobian_parameters_vectors_second_order[i],
    #             solver.linear_solver,
    #             )
    #     else
    #         search_direction_nonsymmetric!(
    #             solver.data.solution_sensitivity_vectors[i],
    #             solver.data.jacobian_variables,
    #             solver.data.jacobian_parameters_vectors[i],
    #             solver.lu_factorization)
    #     end
    #
    #     for (k, s) in enumerate(solver.data.solution_sensitivity_vectors[i].all)
    #         solver.data.solution_sensitivity[k, i] = -1.0 * s
    #     end
    # end

    return
end

A = rand(5,5)
view(A, 1:2, :)

# data.residual.cone_product + S * data.jacobian_parameters[indices.duals, :]
#
a = [1,2,3,4,5]
A = zeros(5,10)

a .+ A
hcat([a for i=1:10]...)
