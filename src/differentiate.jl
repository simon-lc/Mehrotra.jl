function differentiate!(solver; keywords=keys(solver.indices.parameter_keywords))
    data = solver.data
    problem = solver.problem
    methods = solver.methods
    cone_methods = solver.cone_methods
    indices = solver.indices
    dimensions = solver.dimensions
    solution = solver.solution
    parameters = solver.parameters
    options = solver.options
    compressed = options.compressed_search_direction
    sparse_solver = options.sparse_solver

    Mehrotra.evaluate!(problem,
        methods,
        cone_methods,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        equality_jacobian_keywords=keywords,
        cone_constraint=true,
        cone_jacobian=true,
        sparse_solver=sparse_solver,
        compressed=compressed,
    )

    residual!(data, problem, indices,
        residual=true,
        jacobian_variables=true,
        jacobian_parameters=true,
        compressed=compressed,
        sparse_solver=sparse_solver)
    # correction is not needed since it only affect the vector and not the jacobian

    # compute solution sensitivities
    fill!(solver.data.solution_sensitivity, 0.0)
    # data.solution_sensitivity .= - data.jacobian_variables \ data.jacobian_parameters #TODO

    if compressed
        jacobian_variables_compressed = options.sparse_solver ?
            data.jacobian_variables_compressed_sparse : data.jacobian_variables_compressed_dense
        # primal dual step
        linear_solve!(solver.linear_solver,
            view(data.solution_sensitivity, 1:dimensions.equality, :),
            jacobian_variables_compressed,
            view(data.jacobian_parameters, 1:dimensions.equality, :),
            fact=true)
        data.solution_sensitivity .*= -1.0

        # slack step
        for i = 1:dimensions.parameters
            methods.slack_direction(
                view(data.solution_sensitivity, indices.slacks, i), # Δs
                view(data.solution_sensitivity, indices.duals, i), # Δz
                solution.all, # x
                view(data.jacobian_parameters, indices.cone_product, i), # rs
                )
        end
    else
        jacobian_variables = options.sparse_solver ? data.jacobian_variables_sparse.matrix :
            data.jacobian_variables_dense
        linear_solve!(solver.linear_solver, data.solution_sensitivity,
            jacobian_variables, data.jacobian_parameters, fact=true)
        data.solution_sensitivity .*= -1.0
    end
    #TODO parallelize, make more efficient

    # set the state of the solver to differentiated = true
    for k in keywords
        solver.consistency.differentiated[k] = true
    end
    return
end
