using Mehrotra
using MeshCat

include("block_2d_utils.jl")

vis = Visualizer()
render(vis)

function residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, side = unpack_parameters(parameters)

    # velocity
    v25 = y[1:2]
    ω25 = y[3:3]
    p1 = p2 - timestep * v15
    θ1 = θ2 - timestep * ω15
    p3 = p2 + timestep * v25
    θ3 = θ2 + timestep * ω25

    # signed distance function
    ϕ = signed_distance_function(p3, θ3, side)

    γ = z[1:4]
    ψ = z[5:8]
    β = z[9:16]

    sγ = s[1:4]
    sψ = s[5:8]
    sβ = s[9:16]

    N = impact_jacobian(θ3, side)

    D = friction_jacobian(θ3, side)
    P = [
        +D[1:1,:];
        -D[1:1,:];
        +D[2:2,:];
        -D[2:2,:];
        +D[3:3,:];
        -D[3:3,:];
        +D[4:4,:];
        -D[4:4,:];
         ]

    # mass matrix
    M = Diagonal([mass; mass; inertia])

    # friction cone
    fric = [
        friction_coefficient * γ[1:1] - [sum(β[1:2])];
        friction_coefficient * γ[2:2] - [sum(β[3:4])];
        friction_coefficient * γ[3:3] - [sum(β[5:6])];
        friction_coefficient * γ[4:4] - [sum(β[7:8])];
        ]

    # maximum dissipation principle
    mdp = [
        (P[1:2,:] * [v25; ω25] + ψ[1]*ones(2));
        (P[3:4,:] * [v25; ω25] + ψ[2]*ones(2));
        (P[5:6,:] * [v25; ω25] + ψ[3]*ones(2));
        (P[7:8,:] * [v25; ω25] + ψ[4]*ones(2));
        ]

    res = [
        M * ([p3; θ3] - 2*[p2; θ2] + [p1; θ1])/timestep - timestep * [0; mass * gravity; 0] - N'*γ - P'*β - u*timestep;
        sγ - ϕ;
        sψ - fric;
        sβ - mdp;
        ]
    return res
end

function full_residual(solution, parameters, κ)
    primals = solution[1:3]
    duals = solution[3 .+ (1:16)]
    slacks = solution[3+16 .+ (1:16)]
    y, z, s = primals, duals, slacks
    p2, θ2, v15, ω15, u, timestep, mass, inertia, gravity, friction_coefficient, side = unpack_parameters(parameters)

    # velocity
    v25 = y[1:2]
    ω25 = y[3:3]
    p1 = p2 - timestep * v15
    θ1 = θ2 - timestep * ω15
    p3 = p2 + timestep * v25
    θ3 = θ2 + timestep * ω25

    # signed distance function
    ϕ = signed_distance_function(p3, θ3, side)

    γ = z[1:4]
    ψ = z[5:8]
    β = z[9:16]

    sγ = s[1:4]
    sψ = s[5:8]
    sβ = s[9:16]

    N = impact_jacobian(θ3, side)

    D = friction_jacobian(θ3, side)
    P = [
        +D[1:1,:];
        -D[1:1,:];
        +D[2:2,:];
        -D[2:2,:];
        +D[3:3,:];
        -D[3:3,:];
        +D[4:4,:];
        -D[4:4,:];
         ]

    # mass matrix
    M = Diagonal([mass; mass; inertia])

    # friction cone
    fric = [
        friction_coefficient * γ[1:1] - [sum(β[1:2])];
        friction_coefficient * γ[2:2] - [sum(β[3:4])];
        friction_coefficient * γ[3:3] - [sum(β[5:6])];
        friction_coefficient * γ[4:4] - [sum(β[7:8])];
        ]

    # maximum dissipation principle
    mdp = [
        (P[1:2,:] * [v25; ω25] + ψ[1]*ones(2));
        (P[3:4,:] * [v25; ω25] + ψ[2]*ones(2));
        (P[5:6,:] * [v25; ω25] + ψ[3]*ones(2));
        (P[7:8,:] * [v25; ω25] + ψ[4]*ones(2));
        ]

    res = [
        M * ([p3; θ3] - 2*[p2; θ2] + [p1; θ1])/timestep - timestep * [0; mass * gravity; 0] - N'*γ - P'*β - u*timestep;
        sγ - ϕ;
        sψ - fric;
        sβ - mdp;
        z .* s .- κ;
        # z .* s;
        ]
    return res
end

num_primals = 3
num_cone = 16
num_parameters = 15
idx_nn = collect(1:16)
idx_soc = [collect(1:0)]
p2 = [1,1.0]
θ2 = [0.0]
v15 = [0,-1]
ω15 = [0]
u = rand(3)
side = 0.5
timestep = 0.05
friction_coefficient = 1.0
parameters = [p2; θ2; v15; ω15; u; timestep; 1.0; 0.1; -9.81; friction_coefficient; side]

solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(max_iterations=30, verbose=true)
    )

solve!(solver)

variables = solver.solution.all

# no warm starting
H = 100
p2 = [1,1.0]
θ2 = [0.0]
v15 = [-3.0,0.0]
ω15 = [20.0]
u = [0.3*rand(3) for i=1:H]
p, θ, v, ω, iterations = simulate_block_2d(solver, p2, θ2, v15, ω15, u,
    friction_coefficient=friction_coefficient,
    timestep=timestep)


# with warm starting
H = 100
p2 = [1,1.0]
θ2 = [0.0]
v15 = [-3.0,0.0]
ω15 = [20.0]
p, θ, v, ω, warm_iterations = warm_simulate_block_2d(solver, p2, θ2, v15, ω15, u,
    friction_coefficient=friction_coefficient,
    timestep=timestep)


# with sensitivity based warm starting
H = 100
p2 = [1,1.0]
θ2 = [0.0]
v15 = [-3.0,0.0]
ω15 = [20.0]
p, θ, v, ω, sensi_iterations = sensitivity_simulate_block_2d(solver, p2, θ2, v15, ω15, u,
    friction_coefficient=friction_coefficient,
    timestep=timestep)


# with sensitivity based warm starting
H = 1
p2 = [1,1.0]
θ2 = [0.0]
v15 = [-3.0,0.0]
ω15 = [20.0]
p, θ, v, ω, al_iterations = al_simulate_block_2d(solver, p2, θ2, v15, ω15, u,
    friction_coefficient=friction_coefficient,
    timestep=timestep)


scatter(sensi_iterations, ylims=(0,Inf), color=:yellow,
    markershape=:square, markersize=9, label="sensitivity based warm start")
scatter!(warm_iterations, ylims=(0,Inf), color=:red,
    markershape=:star5, markersize=7, label="warm start")
scatter!(iterations, ylims=(0,Inf), color=:black,
    markershape=:circle, markersize=9, label="cold start")

mean(iterations)
mean(warm_iterations)
mean(sensi_iterations)

solver
solver
solver

# plot(hcat(p...)')
render(vis)
setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.compose(
            MeshCat.Translation(p[i][1], 0.0, p[i][2]),
            MeshCat.LinearMap(RotY(θ[i][1])),
            ))
    end
end
MeshCat.setanimation!(vis, anim)









#
#
# p3 = [1,2.0]
# θ3 = [0.01]
# J0 = FiniteDiff.finite_difference_jacobian(pθ -> signed_distance_function(pθ[1:2], pθ[3:3], side), [p3; θ3])
# J1 = FiniteDiff.finite_difference_jacobian(pθ -> tangential_distance_function(pθ[1:2], pθ[3:3], side), [p3; θ3])
# J2 = impact_jacobian(θ3, side)
# J3 = friction_jacobian(θ3, side)
#
# J0 - J2
# J1 - J3
#
