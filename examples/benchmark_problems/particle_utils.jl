function unpack_particle_parameters(parameters)
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

function linear_particle_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_particle_parameters(parameters)

    v25 = y
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    vtan = v25[1:2]

    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:6]

    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:6]

    N = [0 0 1]
    D = [1 0 0;
         0 1 0]
    P = [+D;
         -D]

    res = [
        mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - P' * β - u * timestep;
        sγ - (p3[3:3] .- side/2);
        sψ - (friction_coefficient * γ - [sum(β)]);
        sβ - (P * v25 + ψ[1]*ones(4));
        # z .* s .- κ[1];
        ]
    return res
end

function non_linear_particle_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_particle_parameters(parameters)

    v25 = y
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    γ = z[1:1]
    β = z[2:4]

    sγ = s[1:1]
    sβ = s[2:4]

    N = [0 0 1]
    D = [1 0 0;
         0 1 0]

    vtan = D * v25

    res = [
        mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - D' * β[2:3] - u * timestep;
        sγ - (p3[3:3] .- side/2);
        sβ[2:3] - vtan;
        β[1:1] - friction_coefficient * γ;
        # z ∘ s .- κ[1];
        ]
    return res
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
