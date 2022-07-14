using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots


include("../polytope.jl")
include("../visuals.jl")
include("../rotate.jl")
include("../quaternion.jl")

vis = Visualizer()
render(vis)

include("node.jl")
include("body.jl")
include("contact.jl")
include("mechanism.jl")



################################################################################
# demo
################################################################################
# parameters
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2);
bp = 0.5*[
    +1,
    +1,
    +1,
     1,
     # 2,
    ];
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.00ones(4,2);
bc = 2.0*[
     1,
     1,
     1,
     1,
    ];

timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

# nodes
pbody = Body175(timestep, mass, inertia, [Ap], [bp], gravity=+gravity, name=:pbody);
cbody = Body175(timestep, 1e1*mass, 1e1*inertia, [Ac], [bc], gravity=-0*gravity, name=:cbody);
bodies = [pbody, cbody];
contacts = [Contact175(bodies[1], bodies[2], friction_coefficient=0.3)]
indexing!([bodies; contacts])

# bodies[1].node_index
# bodies[2].node_index
# contacts[1].node_index



function mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)
    num_duals = length(duals)
    num_primals = length(primals)
    num_equality = num_primals + num_duals

    e = zeros(num_equality)
    x = [primals; duals; slacks]
    θ = parameters

    # body
    for body in bodies
        body_residual!(e, x, θ, body)
    end

    # contact
    for contact in contacts
        pbody = find_body(bodies, contact.parent_name)
        cbody = find_body(bodies, contact.child_name)
        contact_residual!(e, x, θ, contact, pbody, cbody)
    end
    return e
end

nodes = [bodies; contacts];
num_primals = sum(primal_dimension.(nodes))
num_cone = sum(cone_dimension.(nodes))
primals = ones(num_primals);
duals = ones(num_cone);
slacks = ones(num_cone);
parameters = vcat(get_parameters.(nodes)...);
mechanism_residual(primals, duals, slacks, parameters, bodies, contacts);

local_mechanism_residual(primals, duals, slacks, parameters) = mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

solver = Solver(
    local_mechanism_residual,
    num_primals,
    num_cone,
    parameters=parameters,
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    method_type=:finite_difference,
    options=Options(
        verbose=false,#true, 
        complementarity_tolerance=1e-3,
        compressed_search_direction=false, 
        sparse_solver=false,
        warm_start=false,
    ));

solve!(solver)

################################################################################
# test simulation
################################################################################
Xp2 = [[+0.1,5.0,1.0]]
Xc2 = [[-0,1.0,-1.0]]
Vp15 = [[-0,0,-0.0]]
Vc15 = [[+0,0,0.0]]
C = []
Np = []
Nc = []
Tp = []
Tc = []
iter = []

H = 35
Up = [zeros(3) for i=1:H]
Uc = [zeros(3) for i=1:H]
for i = 1:H
    bodies[1].pose .= Xp2[end]
    bodies[1].velocity .= Vp15[end]
    bodies[1].input .= Up[i]

    bodies[2].pose .= Xc2[end]
    bodies[2].velocity .= Vc15[end]
    bodies[2].input .= Uc[i]

    θb1 = get_parameters(bodies[1])
    θb2 = get_parameters(bodies[2])
    θc1 = get_parameters(contacts[1])
    parameters .= [θb1; θb2; θc1]
    solver.parameters .= [θb1; θb2; θc1]

    solve!(solver)
    x = deepcopy(solver.solution.all)
    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contacts[1].node_index.x], contacts[1])
    vp25 = unpack_variables(x[bodies[1].node_index.x], bodies[1])
    vc25 = unpack_variables(x[bodies[2].node_index.x], bodies[2])

    push!(Vp15, vp25)
    push!(Vc15, vc25)
    push!(Xp2, Xp2[end] + timestep * vp25)
    push!(Xc2, Xc2[end] + timestep * vc25)

    normal_pw = -x_2d_rotation(Xp2[end][3:3]) * Ap' * λp
    normal_cw = +x_2d_rotation(Xp2[end][3:3]) * Ac' * λc
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    push!(C, c + (Xp2[end][1:2] + Xc2[end][1:2]) ./ 2)
    push!(Np, normal_pw)
    push!(Nc, normal_cw)
    push!(Tp, tangent_pw)
    push!(Tc, tangent_cw)
    push!(iter, solver.trace.iterations)
end


scatter(iter)



################################################################################
# visualization
################################################################################
render(vis)
set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polytope!(vis, Ap, bp, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ac, bc, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

build_rope(vis; N=1, color=Colors.RGBA(0,0,0,1),
    rope_type=:cylinder, rope_radius=0.04, name=:normal)

build_rope(vis; N=1, color=Colors.RGBA(1,0,0,1),
    rope_type=:cylinder, rope_radius=0.04, name=:tangent)

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_straight_rope(vis, [0; C[i]], [0; C[i]+Nw[i]]; N=1, name=:normal)
        set_straight_rope(vis, [0; C[i]], [0; C[i]+Tw[i]]; N=1, name=:tangent)
        set_2d_polytope!(vis, Xp2[i][1:2], Xp2[i][3:3], name=:polya)
        set_2d_polytope!(vis, Xc2[i][1:2], Xc2[i][3:3], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, C[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
# open(vis)
# convert_frames_to_video_and_gif("single_level_hard_offset")
# convert_frames_to_video_and_gif("single_level_hard_tilted")
# convert_frames_to_video_and_gif("single_level_hard")

ex = solver.data.jacobian_variables_dense
# plot(Gray.(abs.(ex)))
# plot(Gray.(abs.(ex - ex')))
# plot(Gray.(abs.(ex + ex')))
# plot(Gray.(1e3abs.(solver.data.jacobian_variables_dense)))

ex[1:6,1:6]
ex[7:9,7:9]

# scatter(solver.solution.all)
# scatter(solver.solution.primals)
# scatter(solver.solution.duals)
# scatter(solver.solution.slacks)

# plot(hcat(solutions...)', legend=false)