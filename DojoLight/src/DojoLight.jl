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
include("shape.jl")

include("dynamics/body.jl")
include("dynamics/poly_halfspace.jl")
include("dynamics/poly_poly.jl")
include("dynamics/poly_sphere.jl")
include("dynamics/sphere_halfspace.jl")
include("dynamics/sphere_sphere.jl")

include("quasistatic_dynamics/robot.jl")
include("quasistatic_dynamics/object.jl")

include("mechanism.jl")
include("contact_frame.jl")
include("storage.jl")
include("simulate.jl")
include("visuals.jl")
include("dynamics.jl")
include("continuation.jl")

include("../environment/polytope_bundle.jl")
include("../environment/polytope_drop.jl")
