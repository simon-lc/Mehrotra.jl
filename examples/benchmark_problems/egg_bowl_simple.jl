using Mehrotra
using MeshCat
using StaticArrays
using Plots
using Random


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
    bowl_radius = parameters[15]
    egg_principal_axes = parameters[16:17]
    return p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, bowl_radius, egg_principal_axes
end

function linear_egg_bowl_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, bowl_radius, egg_principal_axes =
        unpack_egg_bowl_parameters(parameters)

    # velocity
    v25 = y[1:2]
    ω25 = y[3:3]
    # cp = y[4:5]
    p1 = p2 - timestep * v15
    θ1 = θ2 - timestep * ω15
    p3 = p2 + timestep * v25
    θ3 = θ2 + timestep * ω25

    # signed distance function
    # the center of the bowl is in 0,0
    # ϕ = [bowl_radius - norm(cp - [0, 0])]
    # cn = -cp #/ (1e-6 + norm(cp)) # contact normal
    cn = -p3 / (1e-6 + norm(p3)) # contact normal
    cp = p3 - cn * egg_principal_axes[1]
    ϕ = [bowl_radius - norm(cp)]

    R = [0 -1; 1 0]
    ct = R * cn # contact tangent

    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:4]
    # γc = z[5:5]

    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:4]
    # sc = s[5:5]

    N = [cn[1] cn[2] +cross([p3 - cp; 0.0], [cn; 0.0])[3]]
    P = [
        +ct[1] +ct[2] +cross([p3 - cp; 0.0], [ct; 0.0])[3];
        -ct[1] -ct[2] -cross([p3 - cp; 0.0], [ct; 0.0])[3];
    ]

    # mass matrix
    M = Diagonal([mass; mass; inertia])

    # friction cone
    fric = [
        friction_coefficient * γ - [sum(β)];
        ]

    # maximum dissipation principle
    mdp = [
        (P * [v25; ω25] + ψ[1]*ones(2));
        # (v25 + 1*cross([p3 - cp; 0], [0; 0; ω25])[1:2] + ψ[1]*ones(2));
        ]

    res = [
        M * ([p3; θ3] - 2*[p2; θ2] + [p1; θ1])/timestep - timestep * [0; mass * gravity; 0] - N'*γ - P'*β;# - u*timestep;
        sγ - ϕ;
        sψ - fric;
        sβ - mdp;
        # -(cp - [0,0]) + γc .* 2(cp - p3);
        # sc .+ (cp - p3)' * (cp - p3) .- 2*object_principal_axes[1:1].^2;
        ]
    return res
end

# N = [0 0 1]
# D = [1 0 0;
#      0 1 0]
# P = [+D;
#      -D]
#
# res = [
#     mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - P' * β - u * timestep;
#     sγ - (p3[3:3] .- side/2);
#     sψ - (friction_coefficient * γ - [sum(β)]);
#     sβ - (P * v25 + ψ[1]*ones(4));
#     # z .* s .- κ[1];
#     ]
# return res
# end


linear_egg_bowl_residual(rand(num_primals), rand(num_cone), rand(num_cone), parameters)

function simulate_egg_bowl(solver, p2, θ2, v15, ω15, u; timestep=0.01, mass=1.0,
        inertia=0.1, friction_coefficient=0.2, gravity=-9.81, bowl_radius=1.5,
        egg_principal_axes=[0.1, 0.1], warm_start::Bool=false)

    solver.options.verbose = false
    solver.options.warm_start = warm_start

    H = length(u)
    p = []
    θ = []
    v = []
    ω = []
    iterations = Vector{Int}()
    guess = deepcopy(solver.solution.all)

    for i = 1:H
        @show i
        push!(p, p2)
        push!(θ, θ2)
        push!(v, v15)
        push!(ω, ω15)
        parameters = [p2; θ2; v15; ω15; u[i]; timestep; mass; inertia; gravity; friction_coefficient; bowl_radius; egg_principal_axes]
        solver.parameters .= parameters

        warm_start && (solver.solution.all .= guess)
        solve!(solver)
        guess = deepcopy(solver.solution.all)

        push!(iterations, solver.trace.iterations)

        v15 .= solver.solution.primals[1:2]
        ω15 .= solver.solution.primals[3:3]
        p2 = p2 + timestep * v15
        θ2 = θ2 + timestep * ω15
    end
    return p, θ, v, ω, iterations
end


# include("egg_block_utils.jl")

# vis = Visualizer()
# open(vis)

# dimensions and indices
num_primals = 3
num_cone = 4
num_parameters = 17
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

# parameters
p2 = [1,0.0]
θ2 = [0.0]
v15 = [0.0, +1.0]
ω15 = [0.0]
u = [0.4, 0.8, 0.9]
timestep = 0.10
mass = 1.0
inertia = 0.1
gravity = -9.81
friction_coefficient = 2.0
bowl_radius = 1.5
egg_principal_axes = [0.3, 0.3]
parameters = [p2; θ2; v15; ω15; u; timestep; inertia; mass; gravity;
    friction_coefficient; bowl_radius; egg_principal_axes]

################################################################################
# solve
################################################################################
solver = Solver(linear_egg_bowl_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(
        max_iterations=30,
        verbose=true,
        compressed_search_direction=true,
        sparse_solver=false,
        )
    )

solver.solution.all
solver.solution.primals .= rand(num_primals)
solver.solution.duals .= 1e-1*ones(num_cone)
solver.solution.slacks .= 1e-1*ones(num_cone)

solve!(solver)
solver
# Main.@profiler [solve!(solver) for i=1:300]

# @benchmark $solve!($solver)

H = 1500
U = [zeros(3) for i = 1:H]
p, θ, _, _, iterations = simulate_egg_bowl(solver, p2, θ2, v15, ω15, U; timestep=timestep, mass=1.0,
        inertia=inertia,
        friction_coefficient=friction_coefficient,
        gravity=gravity,
        bowl_radius=bowl_radius,
        egg_principal_axes=egg_principal_axes,
        warm_start=false)
plot(iterations)
plot([i[1] for i in θ])


################################################################################
# visualization
################################################################################
using RobotVisualizer
using Meshing

function set_surface!(vis::Visualizer, f::Any;
    xlims=[-20.0, 20.0],
    ylims=[-20.0, 20.0],
    color=RGBA(1.0, 1.0, 1.0, 1.0),
    n::Int=200)
    mesh = GeometryBasics.Mesh(f,
        HyperRectangle(Vec(xlims[1], ylims[1], -2.0), Vec(xlims[2] - xlims[1], ylims[2] - ylims[1], 4.0)),
        Meshing.MarchingCubes(), samples=(n, n, Int(floor(n / 8))))
    setobject!(vis["surface"], mesh, MeshPhongMaterial(color=color))
    return nothing
end


function bowl(x, radius)
    return x[3] + sqrt(max(0, radius^2 - x[1:2]'*x[1:2]))
end


set_light!(vis)
set_background!(vis)
set_floor!(vis[:left], x=2, y=5, origin=[2.5,0,0.0])
set_floor!(vis, x=2, y=5, origin=[-2.5,0,0.0])
set_surface!(vis, x -> bowl(x, bowl_radius), xlims=[-1.5,1.5], ylims=[-0.1, 0.1], color=RGBA(0.5,0.5,0.5,1.0), n=200)


setobject!(vis[:egg][:white], HyperSphere(MeshCat.Point(0, 0, 0.0),
    object_principal_axes[1]))
setobject!(vis[:egg][:black], HyperSphere(MeshCat.Point(1e-3, 0, 0),
    object_principal_axes[1]),
    MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:egg], MeshCat.compose(
            MeshCat.Translation(SVector{3}(p[i][1], 0.0, p[i][2])),
            MeshCat.LinearMap(RotY(θ[i][1])),
            ))
    end
end
MeshCat.setanimation!(vis, anim)
