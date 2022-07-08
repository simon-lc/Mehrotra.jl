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
bodya = Body171(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body171(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact171(bodies[1], bodies[2])]

contact_solver = ContactSolver(Ap, bp, Ac, bc,
    options=Options(
        complementarity_tolerance=1e-6,
        residual_tolerance=1e-8))
xl = zeros(subvariable_dimension(contacts[1]))
xa = [+0.44,0.0]
qa = [0.0]
xb = [-0.44,0.0]
qb = [-0.00]
θl = pack_lp_parameters(xa, qa, xb, qb, Ap, bp, Ac, bc)

using MeshCat
using Plots
# vis = Visualizer()
render(vis)
build_2d_polytope!(vis, Ap, bp, name=:polyp)
build_2d_polytope!(vis, Ac, bc, name=:polyc, color=RGBA(0.2,0.2,0.2,1))
setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

set_2d_polytope!(vis, xa, qa, name=:polyp)
set_2d_polytope!(vis, xb, qb, name=:polyc)
update_subvariables!(xl, θl, contact_solver)
ϕ = xl[1]
p_parent = xl[2:3]
p_child = xl[4:5]
N = xl[6:8]
∂p_parent = reshape(xl[11 .+ (1:30*2)], (2,30))
∂p_child = reshape(xl[11 + 60  .+ (1:30*2)], (2,30))





settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, p_parent[1], p_parent[2])))


lp_contact_solver(Aa, ba, Ab, bb; d::Int=2,
    options=Options(
        residual_tolerance=1e-10,
        complentarity_tolerance=1e-3,
        differentiate=true,
        compressed_search_direction=false))



bodya = Body171(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body171(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
contacts = [Contact171(bodies[1], bodies[2])]






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
