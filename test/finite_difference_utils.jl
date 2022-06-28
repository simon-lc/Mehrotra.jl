################################################################################
# Finite Difference Utils
################################################################################
function problem_solution(original_solver, parameters)
    solver = deepcopy(original_solver)
    solver.parameters .= parameters
    Mehrotra.solve!(solver)
    return deepcopy(solver.solution.all)
end

function extended_residual(residual, indices, variables, parameters)
    idx_nn = indices.cone_nonnegative
    idx_soc = indices.cone_second_order

    primals = variables[indices.primals]
    duals = variables[indices.duals]
    slacks = variables[indices.slacks]

    [residual(primals, duals, slacks, parameters);
     Mehrotra.cone_product(duals, slacks, idx_nn, idx_soc)]
end

function test_residual_jacobian(solver, residual; mode::Symbol=:variables)
    data = solver.data
    problem = solver.problem
    methods = solver.methods
    cone_methods = solver.cone_methods
    solution = solver.solution
    variables = solver.solution.all
    parameters = solver.parameters
    options = solver.options
    indices = solver.indices
    κ = solver.central_paths

    # find solution
    Mehrotra.solve!(solver)

    # evaluate residuals and jacobians
    Mehrotra.evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
    )
    Mehrotra.residual!(data, problem, indices, solution, parameters, κ.tolerance_central_path)

    # reference
    if mode == :variables
        J0 = data.jacobian_variables
    elseif mode == :parameters
        J0 = data.jacobian_parameters
    end

    # finitediff
    if mode == :variables
        J1 = FiniteDiff.finite_difference_jacobian(
            variables -> extended_residual(residual, indices, variables, parameters),
            variables)
    elseif mode == :parameters
        J1 = FiniteDiff.finite_difference_jacobian(
            parameters -> extended_residual(residual, indices, variables, parameters),
            parameters)
    end

    return norm(J0 - J1, Inf) / max(1, norm(J0, Inf), norm(J1, Inf))
end

function test_solution_sensitivity(solver)
    # force sensitivity computation
    solver.options.differentiate = true

    # find solution
    Mehrotra.solve!(solver)

    # reference
    S0 = solver.data.solution_sensitivity

    # finitediff
    S1 = FiniteDiff.finite_difference_jacobian(
        parameters -> problem_solution(solver, parameters),
        solver.parameters, absstep=1e-10)

    return norm(S0 - S1, Inf) / max(1, norm(S0, Inf), norm(S1, Inf))
end
