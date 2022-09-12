function RobotVisualizer.set_surface!(vis::Visualizer, f::Any;
    xlims=[-20.0, 20.0],
    ylims=[-20.0, 20.0],
    zlims=[-2.0, 4.0],
    color=RGBA(1.0, 1.0, 1.0, 1.0),
    wireframe=false,
    n::Int=100)
    mesh = GeometryBasics.Mesh(f,
        MeshCat.HyperRectangle(
            MeshCat.Vec(xlims[1], ylims[1], zlims[1]),
            MeshCat.Vec(xlims[2] - xlims[1], ylims[2] - ylims[1], zlims[2] - zlims[1])),
        Meshing.MarchingCubes(), samples=(n, n, n))
    setobject!(vis["surface"], mesh, MeshPhongMaterial(color=color, wireframe=wireframe))
    return nothing
end

function unpack_egg_bowl_parameters(parameters)
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
    bowl_diag = parameters[15:16]
    egg_diag = parameters[17:18]
    return p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, bowl_diag, egg_diag
end

function linear_egg_bowl_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, bowl_diag, egg_diag =
        unpack_egg_bowl_parameters(parameters)

    # velocity
    v25 = y[1:2]
    ω25 = y[3:3]
    c = y[4:5]
    λ = y[6:6]

    p1 = p2 - timestep * v15
    θ1 = θ2 - timestep * ω15
    p3 = p2 + timestep * v25
    θ3 = θ2 + timestep * ω25

    bRw = [cos(θ3[1]) sin(θ3[1]); -sin(θ3[1]) cos(θ3[1])]
    B = Diagonal(egg_diag)
    W = Diagonal(bowl_diag)
    # contact point as the minimum of a sum of a convex and a concave function (the sum ahas to be convex)
    cw = c + p3
    cb = bRw * (cw - p3)
    # signed distance function
    ϕ = [cb' * B * cb - 1]

    # contact normal and tangent
    cn = - W * cw
    cn ./= norm(cn) + 1e-6
    R = [0 -1; 1 0]
    ct = R * cn

    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:4]

    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:4]

    N = [cn[1] +cn[2] +cross([-p3 + cw; 0.0], [cn; 0.0])[3]]
    P = [
        +ct[1] +ct[2] +cross([-p3 + cw; 0.0], [ct; 0.0])[3];
        -ct[1] -ct[2] -cross([-p3 + cw; 0.0], [ct; 0.0])[3];
    ]

    # mass matrix
    M = Diagonal([mass; mass; 1*inertia])

    # friction cone
    fric = [
        friction_coefficient * γ - [sum(β)];
        ]

    # maximum dissipation principle
    mdp = [
        P * [v25; ω25] + ψ[1]*ones(2);
        ]

    res = [
        M * ([p3; θ3] - 2*[p2; θ2] + [p1; θ1])/timestep - timestep * [0; mass * gravity; 0] - N'*γ - P'*β - u*timestep;
        bRw' * B * cb + 2λ[1] * W * cw;
        (cw' * W * cw .- 1);

        sγ - ϕ;
        sψ - fric;
        sβ - mdp;
        ]
    return res
end

function simulate_egg_bowl(solver, p2, θ2, v15, ω15, u; timestep=0.01, mass=1.0,
        inertia=0.1, friction_coefficient=0.2, gravity=-9.81, bowl_diag=1.5,
        egg_diag=[0.1, 0.1], warm_start::Bool=false, verbose=false)

    solver.options.verbose = verbose
    solver.options.warm_start = warm_start

    H = length(u)
    p = []
    θ = []
    v = []
    ω = []
    c = []
    iterations = Vector{Int}()
    guess = deepcopy(solver.solution.all)

    for i = 1:H
        @show i
        push!(p, p2)
        push!(θ, θ2)
        push!(v, v15)
        push!(ω, ω15)
        parameters = [p2; θ2; v15; ω15; u[i]; timestep; mass; inertia; gravity; friction_coefficient; bowl_diag; egg_diag]
        solver.parameters .= parameters

        warm_start && (solver.solution.all .= guess)
        solve!(solver)
        guess = deepcopy(solver.solution.all)

        push!(iterations, solver.trace.iterations)
        v15 .= solver.solution.primals[1:2]
        ω15 .= solver.solution.primals[3:3]
        p2 = p2 + timestep * v15
        θ2 = θ2 + timestep * ω15
        push!(c, deepcopy(solver.solution.primals[4:5] + p2))
    end
    return p, θ, v, ω, c, iterations
end
