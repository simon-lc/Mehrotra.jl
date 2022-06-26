using Mehrotra
using MeshCat
using StaticArrays

include("particle_utils.jl")

vis = Visualizer()
render(vis)

# dimensions and indices
num_primals = 3
num_cone = 6
num_parameters = 14
idx_nn = collect(1:6)
idx_soc = [collect(1:0)]

# parameters
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = [0.4, 0.8, 0.9]
timestep = 0.01
mass = 1.0
gravity = -9.81
friction_coefficient = 0.05
side = 0.5

parameters = [p2; v15; u; timestep; mass; gravity; friction_coefficient; side]

################################################################################
# solve
################################################################################
solver = Solver(linear_particle_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(max_iterations=30, verbose=true)
    )
solve!(solver)


################################################################################
# simulation
################################################################################
H = 500
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
U = [0*rand(3) for i=1:H]
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
    MeshCat.atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.Translation(SVector{3}(p[i])))
    end
end
MeshCat.setanimation!(vis, anim)
