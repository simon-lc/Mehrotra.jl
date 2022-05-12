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
    for i = 1:H
        push!(p, p2)
        push!(v, v15)
        parameters = [p2; v15; u[i]; timestep; mass; gravity; friction_coefficient; side]
        solver.parameters .= parameters
        solve!(solver)
        v15 .= solver.solution.primals
        p2 = p2 + timestep * v15
    end
    return p, v
end
