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
    idx = solver.indices
    κ = solver.central_paths
    compressed = options.compressed_search_direction
    sparse_solver = options.sparse_solver

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
        sparse_solver=sparse_solver,
        compressed=compressed,
    )
    Mehrotra.residual!(data, problem, idx,# κ.tolerance_central_path,
        compressed=compressed,
        sparse_solver=sparse_solver,
        )

    # reference
    if mode == :variables
        if compressed && sparse_solver
            J0 = data.jacobian_variables_compressed_sparse
        elseif compressed && !sparse_solver
            J0 = data.jacobian_variables_compressed_dense
        elseif !compressed && sparse_solver
            J0 = data.jacobian_variables_sparse.matrix
        elseif !compressed && !sparse_solver
            J0 = data.jacobian_variables_dense
        end
    elseif mode == :parameters
        J0 = sparse_solver ? data.jacobian_parameters_sparse.matrix : data.jacobian_parameters
    end

    # finitediff
    if mode == :variables
        J1 = FiniteDiff.finite_difference_jacobian(
            variables -> extended_residual(residual, idx, variables, parameters),
            variables)
        if compressed
            J1
            D = J1[idx.slackness, idx.slacks]
            Z = J1[idx.cone_product, idx.slacks]
            S = J1[idx.cone_product, idx.duals]
            @show diag(D)
            @show diag(Z)
            @show diag(S)
            @show variables[idx.duals]
            @show variables[idx.slacks]
            @show -D*inv(Z)*S
            @show -variables[idx.slacks] ./ variables[idx.duals]
            J1[idx.slackness, idx.duals] .+= -D*inv(Z)*S
            (J1 = J1[idx.equality, idx.equality])
        end
    elseif mode == :parameters
        J1 = FiniteDiff.finite_difference_jacobian(
            parameters -> extended_residual(residual, idx, variables, parameters),
            parameters)
    end

    return norm(J0 - J1, Inf) / max(1, norm(J0, Inf), norm(J1, Inf))#, J0, J1
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

    return norm(S0 - S1, Inf) / max(1, norm(S0, Inf), norm(S1, Inf))#, S0, S1
end
