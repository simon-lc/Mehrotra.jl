using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots


vis = Visualizer()
render(vis)

include("../contact_model/lp_2d.jl")
include("../polytope.jl")
include("../visuals.jl")
include("../rotate.jl")
include("../quaternion.jl")

include("node.jl")
include("body.jl")
include("contact.jl")
include("mechanism.jl")


################################################################################
# demo
################################################################################
# parameters
# Ap = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ] .- 0.00ones(4,2)
# bp = 0.5*[
#     +1,
#     +1,
#     +1,
#     1,
#     # 2,
#     ]
Θ = Vector(range(0,2π,length=10))[1:end-1]
Ap = cat([[cos(θ) sin(θ)] for θ in Θ]..., dims=1)
bp = 0.5*ones(9)
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.30ones(4,2)
bc = 5.0*[
     1,
     1,
     1,
     1,
    ]



timestep = 0.05
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
μ = [0.3]

# nodes
bodya = Body171(timestep, mass, inertia, [Ap], [bp], gravity=+gravity, name=:bodya)
bodyb = Body171(timestep, 1e6*mass, 1e6*inertia, [Ac], [bc], gravity=-0*gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Friction171(bodies[1], bodies[2], μ)]
indexing!([bodies; contacts])

contacts[1]

# mechanism
local_residual(primals, duals, slacks, parameters) =
    mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)
mech = Mechanism171(local_residual, bodies, contacts)


################################################################################
# test simulation
################################################################################
mech.solver.options.verbose = false
mech.contacts[1].contact_solver.solver.options.verbose = false
mech.contacts[1].contact_solver.solver.options.complementarity_tolerance = 3e-3
mech.contacts[1].contact_solver.solver.options.residual_tolerance



Xa2 = [[+0.1,3.0,0.5]]
Xb2 = [[+0,-3.0,0.0]]
Va15 = [[-2,0,10.0]]
Vb15 = [[+0,0,0.0]]
Pp = []
Pc = []
iter = []

H = 100
Up = [zeros(3) for i=1:H]
Uc = [zeros(3) for i=1:H]
for i = 1:H
    mech.bodies[1].pose .= Xa2[end]
    mech.bodies[1].velocity .= Va15[end]
    mech.bodies[1].input .= Up[i]

    mech.bodies[2].pose .= Xb2[end]
    mech.bodies[2].velocity .= Vb15[end]
    mech.bodies[2].input .= Uc[i]

    θb1 = get_parameters(mech.bodies[1])
    θb2 = get_parameters(mech.bodies[2])
    θc1 = get_parameters(mech.contacts[1])
    mech.parameters .= [θb1; θb2; θc1]
    mech.solver.parameters .= [θb1; θb2; θc1]

    solve!(mech.solver)
    va25 = deepcopy(mech.solver.solution.all[1:3])
    vb25 = deepcopy(mech.solver.solution.all[4:6])
    push!(Va15, va25)
    push!(Vb15, vb25)
    push!(Xa2, Xa2[end] + timestep * va25)
    push!(Xb2, Xb2[end] + timestep * vb25)

    xl = get_outvariables(mech.contacts[1].contact_solver.solver)
    ϕ, p_parent, p_child, _ = unpack_contact_subvariables(xl, mech.contacts[1])
    push!(Pp, p_parent + (Xa2[end][1:2] + Xb2[end][1:2]) ./ 2)
    push!(Pc, p_child + (Xa2[end][1:2] + Xb2[end][1:2]) ./ 2)
    push!(iter, mech.solver.trace.iterations)
end

# scatter(iter)

################################################################################
# visualization
################################################################################
render(vis)
set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polytope!(vis, Ap, bp, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ac, bc, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_2d_polytope!(vis, Xa2[i][1:2], Xa2[i][3:3], name=:polya)
        set_2d_polytope!(vis, Xb2[i][1:2], Xb2[i][3:3], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Pp[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
# open(vis)
convert_frames_to_video_and_gif("polygone_rolling")