using Mehrotra
using MeshCat

include("particle_utils.jl")

vis = Visualizer()
open(vis)

function unpack_parameters(parameters)
    p2 = parameters[1:3]
    v15 = parameters[4:6]
    u = parameters[7:9]
    timestep = parameters[10]
    mass = parameters[11]
    gravity = parameters[12]
    friction_coefficient = parameters[13]
    side = parameters[14]
    return p2, v15, u, timestep, mass, gravity, friction_coefficient, side
end

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

num_primals = 3
num_cone = 6
num_parameters = 14
idx_nn = collect(1:6)
idx_soc = [collect(1:0)]
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = rand(3)
parameters = [p2; v15; u; 0.01; 1.0; -9.81; 0.5; 0.5]

solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(max_iterations=30, verbose=true)
    )

solve!(solver)

variables = solver.solution.all
r, ∇r = residual_complex(solver, variables)
eigvals(Matrix(∇r[1:3,1:3]))
eigvals(Matrix(∇r[5:9,5:9]))


H = 200
p2 = [1,1,1.0]
v15 = [0,-4,1.0]
u = [zeros(3) for i=1:H]
p, v = simulate_particle(solver, p2, v15, u, friction_coefficient=0.3)

plot(hcat(p...)')

setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))

for i = 1:H
    settransform!(vis[:particle], MeshCat.Translation(p[i]...))
    sleep(0.01)
end
