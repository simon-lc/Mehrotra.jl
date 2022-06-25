using Mehrotra
using MeshCat

include("particle_utils.jl")

vis = Visualizer()
open(vis)

function residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_parameters(parameters)

    v25 = y
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    γ = z[1:1]
    β = z[2:4]

    sγ = s[1:1]
    sβ = s[2:4]

    N = [0 0 1]
    D = [1 0 0;
         0 1 0]

    vtan = D * v25

    res = [
        mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - D' * β[2:3] - u * timestep;
        sγ - (p3[3:3] .- side/2);
        sβ[2:3] - vtan;
        β[1:1] - friction_coefficient * γ;
        # z ∘ s .- κ[1];
        ]
    return res
end

num_primals = 3
num_cone = 4
num_parameters = 14
idx_nn = collect(1:1)
idx_soc = [collect(2:4)]
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = rand(3)
side = 0.5
parameters = [p2; v15; u; 0.01; 1.0; -9.81; 0.0; side]


primals = zeros(num_primals)
duals = ones(num_cone)
slacks = ones(num_cone)
residual(primals, duals, slacks, parameters)

solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(max_iterations=30, verbose=true)
    )

solve!(solver)

variables = solver.solution.all
r, ∇r = residual_complex(solver, variables)

FiniteDiff.finite_difference_jacobian(variables -> residual(
    variables[1:num_primals],
    variables[num_primals .+ (1:num_cone)],
    variables[num_primals + num_cone .+ (1:num_cone)],
    parameters,
    ), variables)



H = 300
p2 = [1,1,1.0]
v15 = [0,-4,1.0]
u = [100(rand(3) .- 0.5) for i=1:H]
p, v = simulate_particle(solver, p2, v15, u, friction_coefficient=0.3)

plot(hcat(p...)')

setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))

for i = 1:H
    settransform!(vis[:particle], MeshCat.Translation(p[i]...))
    sleep(0.01)
end
