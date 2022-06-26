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

function linear_block_2d_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, side = unpack_parameters(parameters)

    # velocity
    v25 = y[1:2]
    ω25 = y[3:3]
    p1 = p2 - timestep * v15
    θ1 = θ2 - timestep * ω15
    p3 = p2 + timestep * v25
    θ3 = θ2 + timestep * ω25

    # signed distance function
    ϕ = signed_distance_function(p3, θ3, side)

    γ = z[1:4]
    ψ = z[5:8]
    β = z[9:16]

    sγ = s[1:4]
    sψ = s[5:8]
    sβ = s[9:16]

    N = impact_jacobian(θ3, side)

    D = friction_jacobian(θ3, side)
    P = [
        +D[1:1,:];
        -D[1:1,:];
        +D[2:2,:];
        -D[2:2,:];
        +D[3:3,:];
        -D[3:3,:];
        +D[4:4,:];
        -D[4:4,:];
         ]

    # mass matrix
    M = Diagonal([mass; mass; inertia])

    # friction cone
    fric = [
        friction_coefficient * γ[1:1] - [sum(β[1:2])];
        friction_coefficient * γ[2:2] - [sum(β[3:4])];
        friction_coefficient * γ[3:3] - [sum(β[5:6])];
        friction_coefficient * γ[4:4] - [sum(β[7:8])];
        ]

    # maximum dissipation principle
    mdp = [
        (P[1:2,:] * [v25; ω25] + ψ[1]*ones(2));
        (P[3:4,:] * [v25; ω25] + ψ[2]*ones(2));
        (P[5:6,:] * [v25; ω25] + ψ[3]*ones(2));
        (P[7:8,:] * [v25; ω25] + ψ[4]*ones(2));
        ]

    res = [
        M * ([p3; θ3] - 2*[p2; θ2] + [p1; θ1])/timestep - timestep * [0; mass * gravity; 0] - N'*γ - P'*β - u*timestep;
        sγ - ϕ;
        sψ - fric;
        sβ - mdp;
        ]
    return res
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
