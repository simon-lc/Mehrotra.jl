using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots
using IterativeLQR
using LinearAlgebra


include("polytope.jl")
include("rotate.jl")
include("quaternion.jl")
include("node.jl")
include("body.jl")
include("poly_poly.jl")
include("poly_halfspace.jl")
include("mechanism.jl")
include("contact_frame.jl")
include("storage.jl")
include("simulate.jl")
include("visuals.jl")
include("dynamics.jl")
include("continuation.jl")

include("../environment/polytope_bundle.jl")
include("../environment/polytope_drop.jl")
