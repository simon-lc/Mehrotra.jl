function unpack_variables(variables, num_primals, num_cone)
    x = variables[1:num_primals]
    y = variables[num_primals .+ (1:num_cone)]
    z = variables[num_primals + num_cone .+ (1:num_cone)]
    return x, y, z
end

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

    vtan = v25[2:3]

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
        # Γ .* S .- κ[1];
        ]
    return res
end

using Mehrotra

num_primals = 3
num_cone = 6
num_parameters = 14
idx_nn = collect(1:6)
idx_soc = [collect(1:0)]
variables = ones(num_variables)
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = rand(3)
parameters = [p2; v15; u; 0.01; 1.0; -9.81; 0.3; 0.5]


#
# function Mehrotra.solve!(solver)
#     # initialize
#     initialize_primals!(solver)
#     initialize_cone!(solver)
#     initialize_slack!(solver)
#     initialize_augmented_lagrangian!(solver)
#
#
#
#     return nothing
# end

dim = Dimensions(num_primals, num_cone, num_parameters;
    nonnegative=num_cone,
    second_order=[0,])
ind = Indices(num_primals, num_cone, num_parameters;
    nonnegative=collect(1:num_cone),
    second_order=[collect(1:0)])
meths = generate_gradients(residual, dim, ind)
prob_methods = ProblemMethods(residual, dim, ind)



solver = Solver(residual, num_primals, num_cone, parameters=parameters)
solve!(solver)

solver.indices
