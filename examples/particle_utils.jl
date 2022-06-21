function unpack_parameters(parameters)
    p2 = parameters[1:3]
    v15 = parameters[4:6]
    u = parameters[7:9]
    timestep = parameters[10]
    mass = parameters[11]
    gravity = parameters[12]
    friction_coefficient = parameters[13]
    side = parameters[14]
    return p2, v15, u, timestep, mass, gravity, friction_coefficient, side
end

function simulate_particle(solver, p2, v15, u; timestep=0.01, mass=1.0,
        friction_coefficient=0.2, gravity=-9.81, side=0.5)
    H = length(u)
    p = []
    v = []
    iterations = Vector{Int}()
    for i = 1:H
        push!(p, p2)
        push!(v, v15)
        parameters = [p2; v15; u[i]; timestep; mass; gravity; friction_coefficient; side]
        solver.parameters .= parameters

        solver.options.verbose = false
        solve!(solver)
        push!(iterations, solver.trace.iterations)

        v15 .= solver.solution.primals
        p2 = p2 + timestep * v15
    end
    return p, v, iterations
end

function warm_simulate_particle(solver, p2, v15, u; timestep=0.01, mass=1.0,
        friction_coefficient=0.2, gravity=-9.81, side=0.5)
    H = length(u)
    p = []
    v = []

    iterations = Vector{Int}()
    for i = 1:H
        push!(p, p2)
        push!(v, v15)
        parameters = [p2; v15; u[i]; timestep; mass; gravity; friction_coefficient; side]
        solver.parameters .= parameters

        # warm start
        guess = solver.solution.all
        initialize!(solver, guess)

        solver.options.verbose = false
        solve!(solver, initialization=(i==1))
        push!(iterations, solver.trace.iterations)

        v15 .= solver.solution.primals
        p2 = p2 + timestep * v15
    end
    return p, v, iterations
end


function sensitivity_simulate_particle(solver, p2, v15, u; timestep=0.01, mass=1.0,
        friction_coefficient=0.2, gravity=-9.81, side=0.5)
    H = length(u)
    p = []
    v = []
    # sensitivity = zeros(solver.dimensions.variables, solver.dimensions.parameters+1)
    sensitivity = zeros(solver.dimensions.variables, solver.dimensions.parameters)

    iterations = Vector{Int}()
    for i = 1:H
        push!(p, p2)
        push!(v, v15)
        parameters = [p2; v15; u[i]; timestep; mass; gravity; friction_coefficient; side]
        # Δparameters = [parameters - solver.parameters; 0.0*solver.options.complementarity_tolerance]
        Δparameters = parameters - solver.parameters
        solver.parameters .= parameters

        # warm start
        dsolution_raw = sensitivity * Δparameters
        # s0 = deepcopy(solver.solution.all)
        # correct_sensitivity_step!(solver, dsolution_raw)
        # s1 = deepcopy(solver.solution.all)
        # @show scn(norm(s0 - s1, Inf))
        # guess = solver.solution.all + correct_sensitivity_step(solver, solver.solution.all, dsolution_raw)
        guess = solver.solution.all + dsolution_raw
        initialize!(solver, guess)
        solver.solution.duals .= max.(solver.solution.duals, solver.options.complementarity_tolerance^2)
        solver.solution.slacks .= max.(solver.solution.slacks, solver.options.complementarity_tolerance^2)


        solver.options.verbose = false
        solve!(solver, initialization=(i==1))
        push!(iterations, solver.trace.iterations)

        # sensitivity
        drdsolution = FiniteDiff.finite_difference_jacobian(solution -> full_residual(solution, solver.parameters, solver.options.complementarity_tolerance), solver.solution.all)
        # drdparameters = FiniteDiff.finite_difference_jacobian(pκ -> full_residual(solver.solution.all, pκ[1:solver.dimensions.parameters], pκ[end:end]), [solver.parameters; solver.options.complementarity_tolerance])
        drdparameters = FiniteDiff.finite_difference_jacobian(parameters -> full_residual(solver.solution.all, parameters, 0.0), solver.parameters)
        sensitivity .= - drdsolution \ drdparameters

        v15 .= solver.solution.primals
        p2 = p2 + timestep * v15
    end
    return p, v, iterations
end


function correct_sensitivity_step(solver, solution, dsolution)
    nx = solver.dimensions.variables
    z = solution[solver.indices.duals]
    s = solution[solver.indices.slacks]

    d = zeros(nx)
    function local_residual(dx)
        dz = dx[solver.indices.duals]
        ds = dx[solver.indices.slacks]
        r = [
            dx - dsolution;
            # z .* s + dz .* s + z .* ds .- solver.options.complementarity_tolerance;
            z .* s + dz .* s + z .* ds + dz .* ds .- 10*solver.options.complementarity_tolerance;
            ]
    end
    for i = 1:10
        r = local_residual(d)
        J = FiniteDiff.finite_difference_jacobian(d -> local_residual(d), d)
        d = d - J \ r
    end
    @show norm(d - dsolution)
    return d
end


function correct_sensitivity_step!(solver, dsolution)
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
    κ = solver.central_path
    τ = solver.fraction_to_boundary

    # options
    options = solver.options
    ###################################################
    ###################################################


    # set search direction
    step.all .= dsolution
    @show scn(norm(dsolution, Inf))

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

    @show scn(affine_step_size)

    # centering
    central_path_candidate = centering(solution, step, affine_step_size, indices)
    central_path_target = max(central_path_candidate, options.complementarity_tolerance)

    ## Corrector step
    residual!(data, problem, indices, solution, [central_path_target])
    search_direction_nonsymmetric!(solver.data.step, solver.data)

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

    @show scn(step_size)

    # violations
    residual!(data, problem, indices, solution, options.complementarity_tolerance) # TODO needs to be only recomputing residual of the cone
    equality_violation = norm(data.residual.equality, Inf)
    cone_product_violation = norm(data.residual.cone_product, Inf)

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
        evaluate!(problem, methods, candidate, parameters,
            equality_constraint=true,
            equality_jacobian_variables=true,
        )
        # evaluate candidate cone product constraint and target
        cone!(problem, cone_methods, candidate,
            product=true,
            jacobian=true,
            target=true
        )

        ## Predictor step
        # residual
        residual!(data, problem, indices, candidate, options.complementarity_tolerance) # TODO needs to be options.complementarity_tolerance

        # violations
        equality_violation_candidate = norm(data.residual.equality, Inf)
        cone_product_violation_candidate = norm(data.residual.cone_product, Inf)

        # Test progress
        if (equality_violation_candidate <= equality_violation ||
            cone_product_violation_candidate <= cone_product_violation)
            @show i
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
end
