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
    sensitivity = zeros(solver.dimensions.variables, solver.dimensions.parameters)

    iterations = Vector{Int}()
    for i = 1:H
        push!(p, p2)
        push!(v, v15)
        parameters = [p2; v15; u[i]; timestep; mass; gravity; friction_coefficient; side]
        Δparameters = parameters - solver.parameters
        solver.parameters .= parameters

        # warm start
        guess = solver.solution.all + sensitivity * Δparameters
        initialize!(solver, guess)
        solver.solution.duals .= max.(solver.solution.duals, solver.options.complementarity_tolerance)
        solver.solution.slacks .= max.(solver.solution.slacks, solver.options.complementarity_tolerance)


        solver.options.verbose = false
        solve!(solver, initialization=(i==1))
        push!(iterations, solver.trace.iterations)

        # sensitivity
        drdsolution = FiniteDiff.finite_difference_jacobian(solution -> full_residual(solution, solver.parameters), solver.solution.all)
        drdparameters = FiniteDiff.finite_difference_jacobian(parameters -> full_residual(solver.solution.all, parameters), solver.parameters)
        sensitivity .= - drdsolution \ drdparameters

        v15 .= solver.solution.primals
        p2 = p2 + timestep * v15
    end
    return p, v, iterations
end
