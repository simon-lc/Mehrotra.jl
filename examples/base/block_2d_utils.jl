function unpack_parameters(parameters)
    p2 = parameters[1:2]
    θ2 = parameters[3:3]
    v15 = parameters[4:5]
    ω15 = parameters[6:6]
    u = parameters[7:9]
    timestep = parameters[10]
    mass = parameters[11]
    inertia = parameters[12]
    gravity = parameters[13]
    friction_coefficient = parameters[14]
    side = parameters[15]
    return p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, side
end

function signed_distance_function(p3, θ3, side)
    # corners
    c1 = side/2 .* [+1, +1, 0]
    c2 = side/2 .* [+1, -1, 0]
    c3 = side/2 .* [-1, +1, 0]
    c4 = side/2 .* [-1, -1, 0]

    # corner positions
    R = [+cos(θ3[1]) +sin(θ3[1]) 0;
         -sin(θ3[1]) +cos(θ3[1]) 0;
          0        0       1]
    cp1 = p3[2] + (R * c1)[2]
    cp2 = p3[2] + (R * c2)[2]
    cp3 = p3[2] + (R * c3)[2]
    cp4 = p3[2] + (R * c4)[2]

    # signed distance function
    ϕ = [cp1; cp2; cp3; cp4]

    return ϕ
end

function tangential_distance_function(p3, θ3, side)
    # corners
    c1 = side/2 .* [+1, +1, 0]
    c2 = side/2 .* [+1, -1, 0]
    c3 = side/2 .* [-1, +1, 0]
    c4 = side/2 .* [-1, -1, 0]

    # corner positions
    R = [+cos(θ3[1]) +sin(θ3[1]) 0;
         -sin(θ3[1]) +cos(θ3[1]) 0;
          0        0       1]
    cp1 = p3[1] + (R * c1)[1]
    cp2 = p3[1] + (R * c2)[1]
    cp3 = p3[1] + (R * c3)[1]
    cp4 = p3[1] + (R * c4)[1]

    # tangential distance function
    ϕ = [cp1; cp2; cp3; cp4]

    return ϕ
end

function impact_jacobian(θ3, side)
    c = cos(θ3[1])
    s = sin(θ3[1])

    N = [
        0 1 1*side/2 * (-c -s)
        0 1 1*side/2 * (-c +s)
        0 1 1*side/2 * (+c -s)
        0 1 1*side/2 * (+c +s)
        ]

    return N
end

function friction_jacobian(θ3, side)
    c = cos(θ3[1])
    s = sin(θ3[1])

    N = [
        1 0 1*side/2 * (+c -s)
        1 0 1*side/2 * (-c -s)
        1 0 1*side/2 * (+c +s)
        1 0 1*side/2 * (-c +s)
        ]

    return N
end

function corner_velocity(p3, θ3, v25, ω25, side)
    # corners
    c1 = side/2 .* [+1, +1, 0]
    c2 = side/2 .* [+1, -1, 0]
    c3 = side/2 .* [-1, +1, 0]
    c4 = side/2 .* [-1, -1, 0]

    # rotation matrix
    R = [+cos(θ3[1]) +sin(θ3[1]) 0;
         -sin(θ3[1]) +cos(θ3[1]) 0;
          0        0       1]

    # velocity
    vel = [
        v25 + cross(R * c1, [0; 0; ω25])[1:2],
        v25 + cross(R * c2, [0; 0; ω25])[1:2],
        v25 + cross(R * c3, [0; 0; ω25])[1:2],
        v25 + cross(R * c4, [0; 0; ω25])[1:2],
        ]
    return vel
end

function simulate_block_2d(solver, p2, θ2, v15, ω15, u; timestep=0.01, mass=1.0,
        inertia=0.1, friction_coefficient=0.2, gravity=-9.81, side=0.5)
    H = length(u)
    p = []
    θ = []
    v = []
    ω = []
    iterations = Vector{Int}()

    for i = 1:H
        push!(p, p2)
        push!(θ, θ2)
        push!(v, v15)
        push!(ω, ω15)
        parameters = [p2; θ2; v15; ω15; u[i]; timestep; mass; inertia; gravity; friction_coefficient; side]
        solver.parameters .= parameters

        solver.options.verbose = false
        solve!(solver)
        push!(iterations, solver.trace.iterations)

        v15 .= solver.solution.primals[1:2]
        ω15 .= solver.solution.primals[3:3]
        p2 = p2 + timestep * v15
        θ2 = θ2 + timestep * ω15
    end
    return p, θ, v, ω, iterations
end

function warm_simulate_block_2d(solver, p2, θ2, v15, ω15, u; timestep=0.01, mass=1.0,
        inertia=0.1, friction_coefficient=0.2, gravity=-9.81, side=0.5)
    H = length(u)
    p = []
    θ = []
    v = []
    ω = []
    iterations = Vector{Int}()

    for i = 1:H
        push!(p, p2)
        push!(θ, θ2)
        push!(v, v15)
        push!(ω, ω15)
        parameters = [p2; θ2; v15; ω15; u[i]; timestep; mass; inertia; gravity; friction_coefficient; side]
        solver.parameters .= parameters

        # warm start
        guess = solver.solution.all
        initialize_variables!(solver, guess)

        solver.options.verbose = false
        solve!(solver, initialization=(i==1))
        push!(iterations, solver.trace.iterations)

        v15 .= solver.solution.primals[1:2]
        ω15 .= solver.solution.primals[3:3]
        p2 = p2 + timestep * v15
        θ2 = θ2 + timestep * ω15
    end
    return p, θ, v, ω, iterations
end

function sensitivity_simulate_block_2d(solver, p2, θ2, v15, ω15, u; timestep=0.01, mass=1.0,
        inertia=0.1, friction_coefficient=0.2, gravity=-9.81, side=0.5)
    H = length(u)
    p = []
    θ = []
    v = []
    ω = []
    nc = solver.dimensions.cone
    iterations = Vector{Int}()
    sensitivity = zeros(solver.dimensions.variables, solver.dimensions.parameters)
    sensitivityκ = zeros(solver.dimensions.variables, solver.dimensions.parameters+solver.dimensions.cone)

    for i = 1:H
        push!(p, p2)
        push!(θ, θ2)
        push!(v, v15)
        push!(ω, ω15)
        parameters = [p2; θ2; v15; ω15; u[i]; timestep; mass; inertia; gravity; friction_coefficient; side]
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
        initialize_variables!(solver, guess)
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


        v15 .= solver.solution.primals[1:2]
        ω15 .= solver.solution.primals[3:3]
        p2 = p2 + timestep * v15
        θ2 = θ2 + timestep * ω15
    end
    return p, θ, v, ω, iterations
end

function correct_sensitivity_step(solver, solution, dsolution)
    nx = solver.dimensions.variables
    nc = solver.dimensions.cone
    z = solution[solver.indices.duals]
    s = solution[solver.indices.slacks]

    zcand = z + dsolution[solver.indices.duals]
    scand = s + dsolution[solver.indices.slacks]
    scaling = max.((zcand .* scand) ./ (z .* s), (z .* s) ./ (zcand .* scand),  dsolution[solver.indices.duals] ./ z,  dsolution[solver.indices.slacks] ./ s)
    @show scaling

    d = zeros(nx)
    function local_residual(dx)
        dz = dx[solver.indices.duals]
        ds = dx[solver.indices.slacks]
        r = [
            1e-2(dx - dsolution);
            # z .* s + dz .* s + z .* ds .- solver.options.complementarity_tolerance;
            z .* s + dz .* s + z .* ds + dz .* ds .- scaling * solver.options.complementarity_tolerance;
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


    # evaluate equality constraint & gradient
    evaluate!(problem, methods, solution, parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
    )
    # evaluate cone product constraint and target
    cone!(problem, cone_methods, solution,
        product=true,
        jacobian=true,
        target=true
    )

    ## Predictor step
    # residual
    residual!(data, problem, indices, solution, 0.0000*κ)

    # search direction
    search_direction_nonsymmetric!(solver.data.step, solver.data)
    @show norm(step.all .- dsolution, Inf)

    # # set search direction
    # step.all .= dsolution
    # @show scn(norm(dsolution, Inf))

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

function al_solve(solution, parameters, κ)
    nx = length(solution)
    λ = zeros(nx)
    y = zeros(nx)
    ls_fail = false
    trace = []

    for k = 1:10
        for i = 1:10
            J = FiniteDiff.finite_difference_jacobian(
                solution -> full_residual(solution, parameters, κ),
                solution)
            r = full_residual(solution, parameters, κ)
            @show norm(r)
            push!(trace, r)
            R = [J' * y; r + κ * (λ − y)]
            M = [+κ^2*I(nx) +J';
                 +J       -κ * I(nx)]
            Δ = -M \ R
            α = 1.0
            for j = 1:10
                r_cand = full_residual(solution + α * Δ[1:nx], parameters, κ)
                # R_cand = [J' * y; r_cand + κ * (λ − y)]
                (norm(r_cand, Inf) <= norm(r, Inf)) && break
                # (norm(R_cand, Inf) <= norm(R, Inf)) && break
                α /= 2
                (j == 20) && (ls_fail = true)
            end
            ls_fail && break
            solution = solution + α * Δ[1:nx]
            y += Δ[nx .+ (1:nx)]
        end
        r = full_residual(solution, parameters, κ)
        λ = λ + 1/κ*r
        κ = max(1e-10, κ * 0.1)
    end
    return solution, trace
end

κ = 1e-4
nx = solver.dimensions.variables
solution = zeros(nx)
mass = 1.0
inertia = 0.1
gravity = -9.81
parameters = [p2; θ2; v15; ω15; u[1]; timestep; mass; inertia; gravity; friction_coefficient; side]
sol, trace = al_solve(solution, parameters, κ)

sol[solver.indices.duals]
sol[solver.indices.slacks]

plot(norm.(trace, Inf), yaxis=:log)

function al_simulate_block_2d(solver, p2, θ2, v15, ω15, u; timestep=0.01, mass=1.0,
        inertia=0.1, friction_coefficient=0.2, gravity=-9.81, side=0.5)
    H = length(u)
    p = []
    θ = []
    v = []
    ω = []
    nc = solver.dimensions.cone
    iterations = Vector{Int}()
    # sensitivity = zeros(solver.dimensions.variables, solver.dimensions.parameters)

    for i = 1:H
        push!(p, p2)
        push!(θ, θ2)
        push!(v, v15)
        push!(ω, ω15)
        parameters = [p2; θ2; v15; ω15; u[i]; timestep; mass; inertia; gravity; friction_coefficient; side]
        # Δparameters = parameters - solver.parameters
        solver.parameters .= parameters

        # warm start
        # dsolution_raw = sensitivity * Δparameters
        # guess = solver.solution.all + dsolution_raw
        # initialize_variables!(solver, guess)
        # solver.solution.duals .= max.(solver.solution.duals, solver.options.complementarity_tolerance^2)
        # solver.solution.slacks .= max.(solver.solution.slacks, solver.options.complementarity_tolerance^2)

        solver.solution.all .= al_solve(solver.solution.all, parameters, solver.options.complementarity_tolerance)

        # solver.options.verbose = false
        # solve!(solver, initialization=(i==1))
        # push!(iterations, solver.trace.iterations)

        # sensitivity
        # drdsolution = FiniteDiff.finite_difference_jacobian(solution -> full_residual(solution, solver.parameters, solver.options.complementarity_tolerance), solver.solution.all)
        # drdparameters = FiniteDiff.finite_difference_jacobian(parameters -> full_residual(solver.solution.all, parameters, 0.0), solver.parameters)
        # sensitivity .= - drdsolution \ drdparameters

        v15 .= solver.solution.primals[1:2]
        ω15 .= solver.solution.primals[3:3]
        p2 = p2 + timestep * v15
        θ2 = θ2 + timestep * ω15
    end
    return p, θ, v, ω, iterations
end


# TODO

# al_solve(solution, parameters, κ)
