using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots

vis = Visualizer()
# open(vis)
render(vis)

include("../src/DojoLight.jl")

include("../environment/polytope_bundle.jl")
include("../environment/polytope_drop.jl")
include("../environment/sphere_drop.jl")
include("../environment/sphere_collision.jl")


################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1);


mech = get_sphere_collision(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.1,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=false,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );

# solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+0.01,1.5,-0.25]
xc2 = [-0.0,0.5,-2.25]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(3)
H0 = 150
# solve!(mech.solver)

@elapsed storage = simulate!(mech, z0, H0)
# Main.@profiler [solve!(mech.solver) for i=1:300]
# @benchmark $solve!($(mech.solver))
# scatter(storage.iterations)

################################################################################
# visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)

build_mechanism!(vis, mech)
# @benchmark $build_mechanism!($vis, $mech)
set_mechanism!(vis, mech, storage, 10)
# @benchmark $set_mechanism!($vis, $mech, $storage, 10)

visualize!(vis, mech, storage, build=false)


# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')
