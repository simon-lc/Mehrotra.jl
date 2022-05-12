using Mehrotra

# problem taken from MCPLIB
# https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.653.5142&rep=rep1&type=pdf
function kojima_shindo(z)
    z1 = z[1]
    z2 = z[2]
    z3 = z[3]
    z4 = z[4]
    f1 = 3z1^2 + 2z1*z2 + 2z2^2 + z3 + 3z4 - 6
    f2 = 2z1^2 + z2^2 + z1 + 10z3 + 2z4 - 2
    f3 = 3z1^2 + z1*z2 + 2z2^2 + 2z3+ 9z4 - 9
    f4 = z1^2 + 3z2^2 + 2z3 + 3z4 - 3
    f = [f1, f2, f3, f4]
    return f
end

function residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks

    f = kojima_shindo(z)

    res = [
        s - f;
        # z .* s .- κ[1];
        ]
    return res
end

num_primals = 0
num_cone = 4
num_parameters = 1
idx_nn = collect(1:4)
idx_soc = [collect(1:0)]
parameters = zeros(num_parameters)
options = Options228(
    max_iterations=30,
    residual_tolerance=1e-2,
    complementarity_tolerance=1e-10)

solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=options
    )

Mehrotra.solve!(solver)

duals = [1,0,3,0.0]
slacks = kojima_shindo(duals)
norm(slacks .* duals, Inf)

duals = [sqrt(6),0,3,0.0]
slacks = kojima_shindo(duals)
norm(slacks .* duals, Inf)
