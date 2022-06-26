using Mehrotra


num_primals = 4
num_cone = 15
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

idx_nn = collect(1:0)
idx_soc = [collect(1:3), collect(4:6), collect(7:9), collect(10:12), collect(13:15)]

As = rand(num_primals, num_primals)
A = As' * As
b = rand(num_primals)
Cs = rand(num_cone, num_cone)
C = Cs * Cs'
d = rand(num_cone)
parameters = [vec(A); b; vec(C); d]

solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    )

solve!(solver)
