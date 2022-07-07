using Mehrotra
using MeshCat
using StaticArrays

include("particle_utils.jl")

vis = Visualizer()
render(vis)

# dimensions and indices
num_primals = 3
num_cone = 4
num_parameters = 14
idx_nn = collect(1:1)
idx_soc = [collect(2:4)]

# parameters
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = [0.4, 0.8, 0.9]
timestep = 0.01
mass = 1.0
gravity = -9.81
friction_coefficient = 0.5
side = 0.5

parameters = [p2; v15; u; timestep; mass; gravity; friction_coefficient; side]

################################################################################
# solve
################################################################################
solver = Solver(non_linear_particle_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(
        max_iterations=30,
        verbose=true,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-4,
        differentiate=false,
        )
    )


################################################################################
# simulation
################################################################################
H = 300
p2 = [1,1,1.0]
v15 = [0,-4,1.0]
U = [0(rand(3) .- 0.5) for i=1:H]
p, v, iterations = simulate_particle(solver, p2, v15, U;
    timestep=timestep,
    mass=mass,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side)


################################################################################
# visualization
################################################################################
setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.Translation(SVector{3}(p[i])))
    end
end
MeshCat.setanimation!(vis, anim)
