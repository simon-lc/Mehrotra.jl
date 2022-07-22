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
        sparse_solver=sparse_solver,
        compressed=compressed,
    )

    residual!(data, problem, indices, compressed=compressed, sparse_solver=sparse_solver)
    # correction is not needed since it only affect the vector and not the jacobian

    # compute solution sensitivities
    fill!(solver.data.solution_sensitivity, 0.0)
    # data.solution_sensitivity .= - data.jacobian_variables \ data.jacobian_parameters #TODO

    if compressed
        jacobian_variables_compressed = options.sparse_solver ?
            data.jacobian_variables_compressed_sparse : data.jacobian_variables_compressed_dense
        jacobian_parameters = options.sparse_solver ?
            data.jacobian_parameters_sparse.matrix : data.jacobian_parameters

        # primal dual step
        # data.jacobian_variables_compressed_dense .= data.jacobian_variables_compressed_sparse
        linear_solve!(solver.linear_solver,
            # data.solution_sensitivity[indices.equality, :],
            view(data.solution_sensitivity, 1:dimensions.equality, :),
            jacobian_variables_compressed,
            # jacobian_parameters[indices.equality, :],
            view(data.jacobian_parameters, 1:dimensions.equality, :),
            fact=true)
        data.solution_sensitivity .*= -1.0

        # slack step
        for i = 1:dimensions.parameters
            methods.slack_direction(
                data.solution_sensitivity[indices.slacks, i], # Δs
                data.solution_sensitivity[indices.duals, i], # Δz
                solution.all, # x
                data.jacobian_parameters[indices.cone_product, i], # rs
                )
            # data.solution_sensitivity[indices.slacks, i] .=
                # -Zi * (data.jacobian_parameters[indices.cone_product, i] + S * data.solution_sensitivity[indices.duals, i]) # -Z⁻¹ (cone_product + S * Δz)
            # end
        end

        # # slack step
        # data.solution_sensitivity[indices.slacks, :] .=
        #     -Zi * (data.jacobian_parameters[indices.cone_product, :] + S * data.solution_sensitivity[indices.duals, :]) # -Z⁻¹ (cone_product + S * Δz)
        #     # -Zi * (data.residual.cone_product .+ (S * data.solution_sensitivity[indices.duals, :])) # -Z⁻¹ (cone_product + S * Δz)
        #     # -Zi * (hcat([data.residual.cone_product for i=1:dimensions.parameters]...) + (S * data.solution_sensitivity[indices.duals, :])) # -Z⁻¹ (cone_product + S * Δz)
    else
        jacobian_variables = options.sparse_solver ? data.jacobian_variables_sparse.matrix : data.jacobian_variables_dense
        jacobian_parameters = options.sparse_solver ? data.jacobian_parameters_sparse.matrix : data.jacobian_parameters
        linear_solve!(solver.linear_solver, data.solution_sensitivity,
            jacobian_variables, jacobian_parameters, fact=true)
        data.solution_sensitivity .*= -1.0
    end
    # #TODO parallelize, make more efficient
    return
end
