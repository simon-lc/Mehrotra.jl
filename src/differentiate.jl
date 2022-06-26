function differentiate!(solver)
    data = solver.data
    problem = solver.problem
    methods = solver.methods
    cone_methods = solver.cone_methods
    indices = solver.indices
    solution = solver.solution
    parameters = solver.parameters
    options = solver.options

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
        cone_jacobian_variables=true,
    )

    # Here we only need to update
    # equality_jacobian_variables
    # equality_jacobian_parameters
    # cone_jacobian_variables
    residual!(data, problem, indices, solution, [options.complementarity_tolerance])

    # # TODO: check if we can use current residual Jacobian w/o recomputing
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
    data.solution_sensitivity .= - data.jacobian_variables \ data.jacobian_parameters #TODO

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
