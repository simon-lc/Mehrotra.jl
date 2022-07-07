using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions


vis = Visualizer()
render(vis)

include("../contact_model/lp_2d.jl")
include("../polytope.jl")
include("../visuals.jl")
include("../rotate.jl")
include("../quaternion.jl")


Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
bp = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bc = 0.5*[
     1,
     1,
     1,
     1,
    ]

timestep = 0.01
gravity = -0.0*9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
bodya = Body170(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body170(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact170(bodies[1], bodies[2])]

contact_solver = ContactSolver(Ap, bp, Ac, bc)
xl = zeros(subvariable_dimension(contacts[1]))
extract_subvariables!(xl, contact_solver.solver)

lp_contact_solver(Aa, ba, Ab, bb; d::Int=2,
    options=Options(
        residual_tolerance=1e-10,
        complentarity_tolerance=1e-3,
        differentiate=true,
        compressed_search_direction=false))



bodya = Body170(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body170(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
contacts = [Contact170(bodies[1], bodies[2])]






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
render(vis)
