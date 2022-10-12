using Mehrotra
using MeshCat
using StaticArrays
using Plots
using Random
using RobotVisualizer
using Meshing
using GeometryBasics
using BenchmarkTools

include("ball_ball_utils.jl")

################################################################################
# visualizer
################################################################################
vis = Visualizer()
# render(vis)
open(vis)
set_light!(vis)
set_background!(vis)
set_floor!(vis)
RobotVisualizer.set_camera!(vis, zoom=3.0)

################################################################################
# dimensions and indices
################################################################################
num_primals = 3
num_cone = 4
num_parameters = 16
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

################################################################################
# parameters
################################################################################
p2 = [0.1, 0.1]
θ2 = [-1.0]
v15 = [0.0, +0.0]
ω15 = [-5.0]
u = [0.0, 0.0, 0.0]
timestep = 0.05
mass = 1.0
inertia = 0.1
gravity = -9.81
friction_coefficient = 0.2
outer_ball_radius = [2.0]
inner_ball_radius = [0.20]
parameters = [p2; θ2; v15; ω15; u; timestep; inertia; mass; gravity;
    friction_coefficient; outer_ball_radius; inner_ball_radius]

# test residual
primals = ones(num_primals)
duals = ones(num_cone)
slacks = ones(num_cone)
linear_ball_ball_residual(primals, duals, slacks, parameters)

################################################################################
# solver
################################################################################
solver = Solver(linear_ball_ball_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    method_type=:finite_difference,
    options=Options(
        max_iterations=20,
        verbose=false,
        residual_tolerance=1e-4,
        complementarity_tolerance=1e-5,
        compressed_search_direction=false,
        sparse_solver=false,
        warm_start=false,
        )
    )

solver.solution.all
solver.solution.primals .= 1e-0*ones(num_primals)
solver.solution.duals .= 1e-1*ones(num_cone)
solver.solution.slacks .= 1e-1*ones(num_cone)
solver.parameters .= parameters

@benchmark solve!(solver)
solver.data

################################################################################
# simulation
################################################################################
H = 100
U = [zeros(3) for i = 1:H]
p, θ, _, _, c, iterations = simulate_ball_ball(solver,
        deepcopy(p2),
        deepcopy(θ2),
        deepcopy(v15),
        deepcopy(ω15),
        U;
        timestep=timestep,
        mass=1.0,
        inertia=inertia,
        friction_coefficient=friction_coefficient,
        gravity=gravity,
        outer_ball_radius=outer_ball_radius,
        inner_ball_radius=inner_ball_radius,
        verbose=true,
        warm_start=false)

plot(iterations)
plot([i[1] for i in θ])

################################################################################
# visualization
################################################################################
fW(x) = (x[1:3]' * Diagonal(1/outer_ball_radius[1]^2 * ones(3)) * x[1:3])[1] - 1
fB(x) = (x[1:3]' * Diagonal(1/inner_ball_radius[1]^2 * ones(3)) * x[1:3])[1] - 1

α_egg = 0.2
α_egg = 1.0
set_surface!(vis[:scene][:bowl], fW,
    xlims=[-2,2], ylims=[-2,2], zlims=[-2,-1], color=RGBA(0.7,0.7,0.7,1), n=100)
set_surface!(vis[:scene][:egg][:grey], fB,
    xlims=[-0.5,0.5], ylims=[-0.5,0.5], zlims=[-0.5,0.01], color=RGBA(0.5,0.5,0.5,α_egg), n=100)
set_surface!(vis[:scene][:egg][:black], fB,
    xlims=[-0.5,0.5], ylims=[-0.5,0.5], zlims=[-0.01,0.5], color=RGBA(0,0,0,α_egg), n=100)

setobject!(vis[:scene][:contact], HyperSphere(MeshCat.Point(0, 0, 0.0), 0.07),
    MeshPhongMaterial(color=RGBA(0,0,0,1.0)))
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:scene], MeshCat.Translation(SVector{3}(0, 0, 2+1e-2)))
        settransform!(vis[:scene][:contact], MeshCat.Translation(SVector{3}(0.0, c[i][1], c[i][2])))
        settransform!(vis[:scene][:egg], MeshCat.compose(
            MeshCat.Translation(SVector{3}(0.0, p[i][1], p[i][2])),
            MeshCat.LinearMap(RotX(-θ[i][1])),
            ))
    end
end
MeshCat.setanimation!(vis, anim)
