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