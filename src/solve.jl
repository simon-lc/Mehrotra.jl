function Mehrotra.solve!(solver)
    # initialize
    solver.trace.iterations = 0
    warm_start = solver.options.warm_start
    # TODO replace with initialize_solver!
    !warm_start && initialize_primals!(solver)
    !warm_start && initialize_duals!(solver)
    !warm_start && initialize_slacks!(solver)
    !warm_start && initialize_interior_point!(solver)

    ϵ = 1e-2
    warm_start && (solver.solution.duals .= solver.solution.duals .+ ϵ)
    warm_start && (solver.solution.slacks .= solver.solution.slacks .+ ϵ)

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
    α = solver.step_sizes
    κ = solver.central_paths
    # τ = solver.fraction_to_boundary

    # options
    options = solver.options
    compressed = options.compressed_search_direction
    decoupling = options.complementarity_decoupling
    sparse_solver = options.sparse_solver
    
    # info
    options.verbose && solver_info(solver)

    # evaluate
    evaluate!(problem, methods, cone_methods, solution, parameters,
        equality_constraint=true,
        cone_constraint=true,
        sparse_solver=sparse_solver,
        compressed=compressed,
    )
    # violation
    equality_violation, cone_product_violation = violation(problem, κ.tolerance_central_path)

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
        # evaluate cone product constraint
        evaluate!(problem, methods, cone_methods, solution, parameters,
            equality_constraint=true,
            equality_jacobian_variables=true,
            cone_constraint=true,
            cone_jacobian=true,
            cone_jacobian_inverse=true,
            sparse_solver=sparse_solver,
            compressed=compressed,
        )

        ## Predictor step
        # residual
        residual!(data, problem, indices,# κ.zero_central_path,
            compressed=compressed,
            sparse_solver=sparse_solver)

        # search direction
        search_direction!(solver)
        # affine line search
        α.affine_step_size .= 1.0
        # cone search duals
        cone_search!(α.affine_step_size, z, Δz,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)
        # cone search slacks
        cone_search!(α.affine_step_size, s, Δs,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)

        # centering
        centering!(κ.target_central_path, solution, step, α.affine_step_size, indices, options=options)

        ## Corrector step
        correction!(data, methods, solution, κ.target_central_path; compressed=compressed)
        search_direction!(solver)

        # line search
        α.step_size .= 1.0
        # cone search duals
        cone_search!(α.step_size, z, Δz,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)
        # cone search slacks
        cone_search!(α.step_size, s, Δs,
            indices.cone_nonnegative, indices.cone_second_order;
            τ_nn=0.9500, τ_soc=0.9500, ϵ=1e-14, decoupling=decoupling)

        # # violations
        # residual!(data, problem, indices, κ.tolerance_central_path,
        #     compressed=compressed,
        #     sparse_solver=sparse_solver) # TODO needs to be only recomputing residual of the cone
        # equality_violation = norm(data.residual.equality, Inf)
        # cone_product_violation = cone_violation(solver)
        # violation
        equality_violation, cone_product_violation = violation(problem, κ.tolerance_central_path)

        for i = 1:options.max_iteration_line_search
            # update candidate
            for i = 1:solver.dimensions.primals
                ŷ[i] = y[i] + minimum(α.step_size) * Δy[i]
            end
            for i = 1:solver.dimensions.duals
                ẑ[i] = z[i] + α.step_size[i] * Δz[i]
            end
            for i = 1:solver.dimensions.slacks
                ŝ[i] = s[i] + α.step_size[i] * Δs[i]
            end

            # evaluate candidate equality constraint
            # evaluate candidate cone product constraint
            evaluate!(problem, methods, cone_methods, candidate, parameters,
                equality_constraint=true,
                cone_constraint=true,
                sparse_solver=sparse_solver,
                compressed=compressed,
            )

            ## Predictor step
            # # residual
            # residual!(data, problem, indices, κ.tolerance_central_path,
            #     compressed=compressed, sparse_solver=sparse_solver) # TODO needs to be options.complementarity_tolerance

            # # violations
            # equality_violation_candidate = norm(data.residual.equality, Inf)
            # cone_product_violation_candidate = cone_violation(solver)
            equality_violation_candidate, cone_product_violation_candidate = violation(problem, κ.tolerance_central_path)

            # Test progress
            if (equality_violation_candidate <= equality_violation ||
                cone_product_violation_candidate <= cone_product_violation)
                break
            end

            # decrease step size
            α.step_size .= options.scaling_line_search .* α.step_size

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
            minimum(α.step_size))
    end

    # failure
    options.verbose && solver_status(solver, false)
    return false
end
