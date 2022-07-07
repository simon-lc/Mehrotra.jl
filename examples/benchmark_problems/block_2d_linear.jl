using Mehrotra
using MeshCat
using StaticArrays
using Plots
using Random

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
friction_coefficient = 0.2
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
# begin
H = 1000
p2 = [1,1.0]
θ2 = [6.0]
v15 = [-5.0,-5.0]*1e-2/timestep
ω15 = [20.0]*1e-2/timestep
Random.seed!(0)
U = [0*5.0*1e-2*[rand(2) .- 0.5; 0] ./ timestep for i=1:H]
solver.options.complementarity_decoupling = false
p, θ, v, ω, cold_iterations = simulate_block_2d(solver, p2, θ2, v15, ω15, U;
    timestep=timestep,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side,
    warm_start=false)

H = 1000
p2 = [1,1.0]
θ2 = [6.0]
v15 = [-5.0,-5.0]*1e-2/timestep
ω15 = [20.0]*1e-2/timestep
Random.seed!(0)
U = [0*5.0*1e-2*[rand(2) .- 0.5; 0] ./ timestep for i=1:H]
solver.options.complementarity_decoupling = true
p, θ, v, ω, cold_iterations_decoupled = simulate_block_2d(solver, p2, θ2, v15, ω15, U;
    timestep=timestep,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side,
    warm_start=false)

H = 1000
p2 = [1,1.0]
θ2 = [6.0]
v15 = [-5.0,-5.0]*1e-2/timestep
ω15 = [20.0]*1e-2/timestep
Random.seed!(0)
U = [0*5.0*1e-2*[rand(2) .- 0.5; 0] ./ timestep for i=1:H]
solver.options.complementarity_decoupling = false
p, θ, v, ω, warm_iterations = simulate_block_2d(solver, p2, θ2, v15, ω15, U;
    timestep=timestep,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side,
    warm_start=true)

H = 1000
p2 = [1,1.0]
θ2 = [6.0]
v15 = [-5.0,-5.0]*1e-2/timestep
ω15 = [20.0]*1e-2/timestep
Random.seed!(0)
U = [0*5.0*1e-2*[rand(2) .- 0.5; 0] ./ timestep for i=1:H]
solver.options.complementarity_decoupling = true
p, θ, v, ω, warm_iterations_decoupled = simulate_block_2d(solver, p2, θ2, v15, ω15, U;
    timestep=timestep,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    gravity=gravity,
    side=side,
    warm_start=true)
# end


scatter(cold_iterations, color=:blue, markersize=7.0, ylims=(0,Inf))
scatter!(cold_iterations_decoupled, color=:blue, markersize=7.0, marker=:square)
scatter!(warm_iterations, color=:red, markersize=7.0)
scatter!(warm_iterations_decoupled, color=:red, markersize=7.0, marker=:square)

scatter(cold_iterations_decoupled - cold_iterations, color=:blue, markersize=7.0, marker=:square)
scatter(warm_iterations_decoupled - warm_iterations, color=:red, markersize=7.0, marker=:square)
mean(cold_iterations)
mean(cold_iterations_decoupled)
mean(warm_iterations)
mean(warm_iterations_decoupled)
mean(warm_iterations_decoupled)


################################################################################
# visualization
################################################################################
setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.compose(
            MeshCat.Translation(SVector{3}(p[i][1], 0.0, p[i][2])),
            MeshCat.LinearMap(RotY(θ[i][1])),
            ))
    end
end
MeshCat.setanimation!(vis, anim)

render(vis)
