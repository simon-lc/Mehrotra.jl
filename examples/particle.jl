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

function residual(variables, parameters)
    x, y, z = unpack_variables(variables, num_primals, num_cone)
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_parameters(parameters)

    v25 = x
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    vtan = v25[2:3]

    sγ = z[1:1]
    sψ = z[2:2]
    sβ = z[3:6]

    γ = y[1:1]
    ψ = y[2:2]
    β = y[3:6]

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

num_primals = 3
num_cone = 6
num_variables = num_primals + 2 * num_cone
num_parameters = 14
idx_nn = collect(1:6)
variables = ones(num_variables)
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = rand(3)
parameters = [p2; v15; u; 0.01; 1.0; -9.81; 0.3; 0.5]


residual(variables, parameters)
generate_gradients(residual, num_variables, num_parameters)
ProblemMethods(num_variables, num_parameters, residual)
generate_cones(num_cone, idx_nn, idx_soc)

dims = Dimensions(num_primals, num_cone, num_parameters)
idx = Indices(num_primals, num_cone, num_parameters)
SolverData(dims, idx)
