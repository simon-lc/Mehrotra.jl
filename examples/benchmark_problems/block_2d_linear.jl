using Mehrotra
using MeshCat
using StaticArrays

include("block_2d_utils.jl")

vis = Visualizer()
render(vis)

# dimensions and indices
num_primals = 3
num_cone = 16
num_parameters = 15
idx_nn = collect(1:16)
idx_soc = [collect(1:0)]

# parameters
p2 = [1,1.0]
θ2 = [0.0]
v15 = [0,-1]
ω15 = [0]
u = [0.4, 0.8, 0.9]
timestep = 0.01
mass = 1.0
inertia = 0.1
gravity = -9.81
friction_coefficient = 0.8
side = 0.5
parameters = [p2; θ2; v15; ω15; u; timestep; inertia; mass; gravity; friction_coefficient; side]


################################################################################
# solve
################################################################################
solver = Solver(linear_block_2d_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(max_iterations=30, verbose=true)
    )
solve!(solver)


################################################################################
# simulation
################################################################################
H = 1000
p2 = [1,1.0]
θ2 = [3.0]
v15 = [-3.0,0.0]
ω15 = [20.0]
U = [zeros(3) for i=1:H]
p, θ, v, ω = simulate_block_2d(solver, p2, θ2, v15, ω15, U;
    timestep=timestep,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side)


################################################################################
# visualization
################################################################################
setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))
anim = MeshCat.Animation(100)

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.compose(
            MeshCat.Translation(SVector{3}(p[i][1], 0.0, p[i][2])),
            MeshCat.LinearMap(RotY(θ[i][1])),
            ))
    end
end
MeshCat.setanimation!(vis, anim)
