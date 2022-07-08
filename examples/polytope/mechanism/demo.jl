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

include("node.jl")
include("body.jl")
include("contact.jl")


################################################################################
# demo
################################################################################
# parameters
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

# nodes
bodya = Body171(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body171(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact171(bodies[1], bodies[2])]
indexing!([bodies; contacts])

# mechanism
local_residual(primals, duals, slacks, parameters) =
    mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)
mech = Mechanism171(local_residual, bodies, contacts)





solve!(mech.solver)
