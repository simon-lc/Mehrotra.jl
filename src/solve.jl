function Mehrotra.solve!(solver; initialization::Bool=true)
    # initialize
    solver.trace.iterations = 0
    initialization && initialize_primals!(solver)
    initialization && initialize_duals!(solver)
    initialization && initialize_slacks!(solver)
    initialization && initialize_interior_point!(solver)

    # indices
    indices = solver.indices

    # variables
    solution = solver.solution
    y = solution.primals
    z = solution.duals
    s = solution.slacks

    # candidate
    candidate = solver.candidate
    ŷ = candidate.primals
    ẑ = candidate.duals
    ŝ = candidate.slacks

    # parameters
    parameters = solver.parameters

    # solver data
    data = solver.data

    # search direction
    step = data.step
    Δy = step.primals
    Δz = step.duals
    Δs = step.slacks

    # problem
    problem = solver.problem
    methods = solver.methods
    cone_methods = solver.cone_methods

    # barrier + augmented Lagrangian
    κ = solver.central_paths
    τ = solver.fraction_to_boundary

    # options
    options = solver.options
    compressed = options.compressed_search_direction

    # info
    options.verbose && solver_info(solver)

    # evaluate
    evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
    )

    # residual
    residual!(data, problem, indices, solution, parameters,
        κ.tolerance_central_path, compressed=compressed)

    # violations
    equality_violation = norm(data.residual.equality, Inf)
    cone_product_violation = cone_violation(solver)

    for i = 1:options.max_iterations
        solver.trace.iterations += 1
        # check for convergence
        if (equality_violation <= options.residual_tolerance &&
            cone_product_violation <= options.residual_tolerance)

            # differentiate
            options.differentiate && differentiate!(solver)

            options.verbose && solver_status(solver, true)
            return true
        end

        # evaluate equality constraint & gradient
        # evaluate cone product constraint and target
        evaluate!(problem, methods, cone_methods, solution, parameters,
            equality_constraint=true,
            equality_jacobian_variables=true,
            cone_constraint=true,
            cone_jacobian=true,
            cone_jacobian_inverse=true,
        )

        ## Predictor step
        # residual
        residual!(data, problem, indices, solution, parameters,
            κ.zero_central_path, compressed=compressed)

        # search direction
        # unstructured_search_direction!(solver)
        search_direction!(solver, compressed=compressed)
        # affine line search
        affine_step_size = 1.0
        # cone search duals
        affine_step_size = cone_search(affine_step_size, z, Δz,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.99, τ_soc=0.99, ϵ=1e-14)
        # cone search slacks
        affine_step_size = cone_search(affine_step_size, s, Δs,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.99, τ_soc=0.99, ϵ=1e-14)

        # centering
        candidate_central_path = centering(solution, step, affine_step_size, indices)
        κ.target_central_path .= max(candidate_central_path, options.complementarity_tolerance)

        ## Corrector step
        residual!(data, problem, indices, solution, parameters, κ.target_central_path)
        # unstructured_search_direction!(solver)
        search_direction!(solver, compressed=compressed)

        # line search
        step_size = 1.0
        # cone search duals
        step_size = cone_search(step_size, z, Δz,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.99, τ_soc=0.99, ϵ=1e-14)
        # cone search slacks
        step_size = cone_search(step_size, s, Δs,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.99, τ_soc=0.99, ϵ=1e-14)

        # violations
        residual!(data, problem, indices, solution, parameters,
            κ.tolerance_central_path, compressed=compressed) # TODO needs to be only recomputing residual of the cone
        equality_violation = norm(data.residual.equality, Inf)
        cone_product_violation = cone_violation(solver)

        for i = 1:options.max_iteration_line_search
            # update candidate
            for i = 1:solver.dimensions.primals
                ŷ[i] = y[i] + step_size * Δy[i]
            end
            for i = 1:solver.dimensions.duals
                ẑ[i] = z[i] + step_size * Δz[i]
            end
            for i = 1:solver.dimensions.slacks
                ŝ[i] = s[i] + step_size * Δs[i]
            end

            # evaluate candidate equality constraint & gradient
            # evaluate candidate cone product constraint and target
            evaluate!(problem, methods, cone_methods, candidate, parameters,
                equality_constraint=true,
                equality_jacobian_variables=true,
                cone_constraint=true,
                cone_jacobian=true,
                cone_jacobian_inverse=true,
            )

            ## Predictor step
            # residual
            residual!(data, problem, indices, candidate, parameters,
                κ.tolerance_central_path, compressed=compressed) # TODO needs to be options.complementarity_tolerance

            # violations
            equality_violation_candidate = norm(data.residual.equality, Inf)
            cone_product_violation_candidate = cone_violation(solver)


            # Test progress
            if (equality_violation_candidate <= equality_violation ||
                cone_product_violation_candidate <= cone_product_violation)
                break
            end

            # decrease step size
            step_size = options.scaling_line_search * step_size

            i == options.max_iteration_line_search && (options.verbose && (@warn "line search failure"); break)
        end

        # update
        for i = 1:solver.dimensions.primals
            y[i] = ŷ[i]
        end
        for i = 1:solver.dimensions.duals
            z[i] = ẑ[i]
        end
        for i = 1:solver.dimensions.slacks
            s[i] = ŝ[i]
        end

        # status
        options.verbose && iteration_status(
            i,
            equality_violation,
            cone_product_violation,
            κ.target_central_path[1],
            step_size)
    end

    # failure
    options.verbose && solver_status(solver, false)
    return false
end
