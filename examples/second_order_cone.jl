using Mehrotra

function unpack_parameters(parameters, num_primals, num_cone)
    off = 0
    A = reshape(parameters[off .+ (1:num_primals^2)], num_primals, num_primals)
    off += num_primals^2
    b = reshape(parameters[off .+ (1:num_primals)], num_primals)
    off += num_primals
    C = reshape(parameters[off .+ (1:num_cone^2)], num_cone, num_cone)
    off += num_cone^2
    b = reshape(parameters[off .+ (1:num_cone)], num_cone)
    off += num_cone
    return A, b, C, d
end

function residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_primals = length(primals)
    num_duals = length(duals)
    A, b, C, d = unpack_parameters(parameters, num_primals, num_cone)

    res = [
        A * y + b;
        C * z + d - s;
        # z .* s .- κ[1];
        ]
    return res
end

num_primals = 4
num_cone = 15
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

idx_nn = collect(1:0)
idx_soc = [collect(1:3), collect(1:3), collect(1:3), collect(1:3), collect(1:3)]

As = rand(num_primals, num_primals)
A = As' * As
Af = reshape(A, num_primals^2)
b = rand(num_primals)
Cs = rand(num_cone, num_cone)
C = Cs - Cs'
Cf = reshape(C, num_cone^2)
d = rand(num_cone)
parameters = [Af; b; Cf; d]

solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    )

solve!(solver)
