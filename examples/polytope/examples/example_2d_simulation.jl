using Plots
using MeshCat
using Polyhedra
using GeometryBasics
using RobotVisualizer
using Quaternions
using StaticArrays


include("polyhedron.jl")
# include("signed_distance.jl")
# include("residual.jl")
include("residual_qp_2d.jl")
include("visuals.jl")
include("quaternion.jl")
include("rotate.jl")
include("../benchmark_problems/block_2d_utils.jl")

vis = Visualizer()
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

# ContactPoint pt;
# pt.world_normal_on_b = normal_on_b;
# pt.world_point_on_a = point_a_world;
# pt.world_point_on_b = point_b_world;
# pt.distance = distance;

################################################################################
# contact
################################################################################
function contact_solver(parameters; na=0, nb=0, d=0,
        options::Options228=Options228(max_iterations=30, verbose=true))

    num_primals = 2d
    num_cone = na + nb

    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    sized_contact_residual(primals, duals, slacks, parameters) = contact_residual(
        primals, duals, slacks, parameters; na=na, nb=nb, d=d)

    solver = Solver(sized_contact_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=options,

        )
    return solver
end

function set_poses!(solver::Solver228, xa, qa, xb, qb)
    d = length(xa)
    off = 0
    solver.parameters[off .+ (1:d)] .= xa; off += d
    solver.parameters[off .+ (1:d)] .= qa; off += 1
    solver.parameters[off .+ (1:d)] .= xb; off += d
    solver.parameters[off .+ (1:d)] .= qb; off += 1
    return nothing
end

function contact_distance(solver, xa, qa, xb, qb)
    d = length(xa)
    set_poses!(solver, xa, qa, xb, qb)
    solve!(solver)

    y = solver.solution.primals
    # ya is expressed in bodya's frame
    ya = y[1:d]
    # y2 is expressed in bodyb's frame
    yb = y[d .+ (1:d)]
    # yw is expressed in world frame
    yaw = xa + x_2d_rotation(qa) * ya
    ybw = xb + x_2d_rotation(qb) * yb
    return norm(yaw - ybw), yaw, ybw
end

# function contact_jacobian(x1, q1, x2, q2, A1, b1, A2, b2, δ)
#     return distance_jacobian_inputs, tangential_velocities, rotational_velocity
# end




A1 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
b1 = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]

A2 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
b2 = 0.5*[
     1,
     1,
     1,
     1,
    ]
δ = 1e2

build_2d_polyhedron!(vis, A1, b1, color=RGBA(0.2,0.2,0.2,0.6), name=:poly1)
build_2d_polyhedron!(vis, A2, b2, color=RGBA(0.8,0.8,0.8,0.6), name=:poly2)

x1 = [1,1.0]
x2 = [0,3.0]
q1 = [+0.5]
q2 = [-0.5]

set_2d_polyhedron!(vis, x1, q1, name=:poly1)
set_2d_polyhedron!(vis, x2, q2, name=:poly2)

parameters = pack_contact_parameters(x1, q1, x2, q2, A1, b1, A2, b2, δ)
solver = contact_solver(parameters, na=size(A1)[1], nb=size(A2)[1], d=size(A1)[2])
distance, y1w, y2w = contact_distance(solver, x1, q1, x2, q2)


solver.options.verbose = false
@benchmark solve!(solver)
Main.@profiler [solve!(solver) for i=1:1000]

setobject!(vis[:contact1],
    HyperSphere(GeometryBasics.Point(0, y1w...), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

setobject!(vis[:contact2],
    HyperSphere(GeometryBasics.Point(0, y2w...), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))





# include("polytope_utils.jl")
################################################################################
# resolve
################################################################################
# dimensions and indices
num_primals = 3
num_cone = 4
num_parameters = 15
idx_nn = collect(1:16)
idx_soc = [collect(1:0)]

# parameters
p2 = [1,1.0]
θ2 = [0.0]
v15 = [0,-1]
ω15 = [0]
u = [0.4, 0.8, 0.9]
timestep = 0.01
mass = 1.0
inertia = 0.1
gravity = -9.81
friction_coefficient = 0.8
side = 0.5
parameters = [p2; θ2; v15; ω15; u; timestep; inertia; mass; gravity; friction_coefficient; side]


################################################################################
# residual
################################################################################

function unpack_polytope_parameters(parameters)
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

function linear_polytope_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, side = unpack_block_2d_parameters(parameters)

    # velocity
    v25 = y[1:2]
    ω25 = y[3:3]
    p1 = p2 - timestep * v15
    θ1 = θ2 - timestep * ω15
    p3 = p2 + timestep * v25
    θ3 = θ2 + timestep * ω25

    # signed distance function
    ϕ = signed_distance_function(p3, θ3, side) ###########################

    γ = z[1:4]
    # ψ = z[5:8]
    # β = z[9:16]

    sγ = s[1:4]
    # sψ = s[5:8]
    # sβ = s[9:16]

    N = impact_jacobian(θ3, side) #########################

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

    # # friction cone
    # fric = [
    #     friction_coefficient * γ[1:1] - [sum(β[1:2])];
    #     friction_coefficient * γ[2:2] - [sum(β[3:4])];
    #     friction_coefficient * γ[3:3] - [sum(β[5:6])];
    #     friction_coefficient * γ[4:4] - [sum(β[7:8])];
    #     ]

    # # maximum dissipation principle
    # mdp = [
    #     (P[1:2,:] * [v25; ω25] + ψ[1]*ones(2));
    #     (P[3:4,:] * [v25; ω25] + ψ[2]*ones(2));
    #     (P[5:6,:] * [v25; ω25] + ψ[3]*ones(2));
    #     (P[7:8,:] * [v25; ω25] + ψ[4]*ones(2));
    #     ]

    res = [
        # M * ([p3; θ3] - 2*[p2; θ2] + [p1; θ1])/timestep - timestep * [0; mass * gravity; 0] - N'*γ - P'*β - u*timestep;
        M * ([p3; θ3] - 2*[p2; θ2] + [p1; θ1])/timestep - timestep * [0; mass * gravity; 0] - N'*γ - u*timestep;
        sγ - ϕ;
        # sψ - fric;
        # sβ - mdp;
        ]
    return res
end

function ProblemMethods(equality::Function, equality_jacobian_variables::Function,
        equality_jacobian_parameters::Function)

    ex_sparsity = collect(zip([findnz(ex)[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(eθ)[1:2]...]...))

    methods = ProblemMethods228(
        equality_constraint,
        equality_jacobian_variables,
        equality_jacobian_parameters,
        zeros(length(ex_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        eθ_sparsity,
    )

    return methods
end

ProblemMethods(e, ex, eθ)





################################################################################
# solve
################################################################################
solver = Solver(linear_block_2d_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(max_iterations=30, verbose=true)
    )
solve!(solver)


################################################################################
# simulation
################################################################################
H = 1000
p2 = [1,1.0]
θ2 = [3.0]
v15 = [-3.0,0.0]
ω15 = [20.0]
U = [zeros(3) for i=1:H]
p, θ, v, ω, iterations = simulate_block_2d(solver, p2, θ2, v15, ω15, U;
    timestep=timestep,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side)


################################################################################
# visualization
################################################################################
setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))
anim = MeshCat.Animation(100)

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.compose(
            MeshCat.Translation(SVector{3}(p[i][1], 0.0, p[i][2])),
            MeshCat.LinearMap(RotY(θ[i][1])),
            ))
    end
end
MeshCat.setanimation!(vis, anim)
