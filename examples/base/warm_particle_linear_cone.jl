using Mehrotra
using MeshCat

include("particle_utils.jl")

vis = Visualizer()
render(vis)

function residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_parameters(parameters)

    v25 = y
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    vtan = v25[1:2]

    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:6]

    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:6]

    N = [0 0 1]
    D = [1 0 0;
         0 1 0]
    P = [+D;
         -D]

    res = [
        mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - P' * β - u * timestep;
        sγ - (p3[3:3] .- side/2);
        sψ - (friction_coefficient * γ - [sum(β)]);
        sβ - (P * v25 + ψ[1]*ones(4));
        # z .* s .- κ[1];
        ]
    return res
end

function full_residual(solution, parameters, κ)
    primals = solution[1:3]
    duals = solution[3 .+ (1:6)]
    slacks = solution[3+6 .+ (1:6)]
    y, z, s = primals, duals, slacks
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_parameters(parameters)

    v25 = y
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    vtan = v25[1:2]

    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:6]

    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:6]

    N = [0 0 1]
    D = [1 0 0;
         0 1 0]
    P = [+D;
         -D]

    res = [
        mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - P' * β - u * timestep;
        sγ - (p3[3:3] .- side/2);
        sψ - (friction_coefficient * γ - [sum(β)]);
        sβ - (P * v25 + ψ[1]*ones(4));
        z .* s .- κ[1];
        # z .* s;
        ]
    return res
end

num_primals = 3
num_cone = 6
num_parameters = 14
idx_nn = collect(1:6)
idx_soc = [collect(1:0)]
p2 = [1,1,1.0]
v15 = [1,-1,1.0]
u = rand(3)
side = 0.5
timestep = 0.15
friction_coefficient = 0.3
parameters = [p2; v15; u; timestep; 1.0; -9.81; friction_coefficient; side]

solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(max_iterations=30, verbose=true)
    )

solve!(solver)
solver.trace.iterations

# no warm starting
H = 100
p2 = [1,1,1.0]
v15 = [2,-4,1.0]
u = [0.3rand(3) for i=1:H]
p, v, iterations = simulate_particle(
    solver, p2, v15, u,
    friction_coefficient=friction_coefficient,
    timestep=timestep)
plot(hcat(p...)')

# with warm starting
H = 100
p2 = [1,1,1.0]
v15 = [2,-4,1.0]
warm_p, warm_v, warm_iterations = warm_simulate_particle(
    solver, p2, v15, u,
    friction_coefficient=friction_coefficient,
    timestep=timestep)

# with sensitivity based warm starting
H = 100
p2 = [1,1,1.0]
v15 = [2,-4,1.0]
sensi_p, sensi_v, sensi_iterations = sensitivity_simulate_particle(
    solver, p2, v15, u,
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


################################################################################
# visualization
################################################################################
render(vis)
setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H
    atframe(anim, i) do
        settransform!(vis[:particle], MeshCat.compose(
            MeshCat.Translation(p[i]...),
            ))
    end
end
MeshCat.setanimation!(vis, anim)
