
# function definition
function f1234(x1::Vector, x11::Vector; symbolic_parsing::Bool=false)
    @rootlayer x1
    @layer x11
    x2 = x1 .+ 1
    x21 = x11 .+ 1
    @layer x2
    @layer x21
    x3 = [x2; x21[1]; x2[2]]
    @layer x3
    x4 = x3 * x3[end] * x3[end-1] * x3[1]
    @layer x4
    x5 = [x4[1]; sin.(x4 * x4[end])]
    @layer x5
    x6 = x5 .+ x5[1] .+ x5[1]
    x7 = x5 .+ x5[1] .+ x5[1]
    @layer x6
    @leaflayer x7
    return x7
end

# root dimensions
n = 2
x1 = rand(n)
x11 = rand(2n)

var_names0, graph0, variables0, expr0, expr_edge0, edge_dict0 = f1234(x1, x11; symbolic_parsing=true)




################################################################################
# test
################################################################################
n = 5
symgraph = generate_symgraph(f1234, [n,n])
xroots0 = [rand(n), rand(n)]
xroot10 = xroots0[1]
xroot20 = xroots0[2]

root0 = :x1
leaf0 = :x7

# evaluation
evaluation50!(symgraph, xroots0)
# Main.@code_warntype evaluation50!(symgraph, xroots0)
# @benchmark $evaluation50!($symgraph, $xroots0)
num_allocs = @ballocated $evaluation50!($symgraph, $xroots0)
e1 = get_evaluation(symgraph, leaf0)
symgraph
e0 = f1234(xroots0...)
norm(e0, Inf)
norm(e1, Inf)
norm(e0 - e1, Inf)


# jacobian vector product
v0 = rand(n+3)
jvp50!(symgraph, v0, xroots0, leaf0)
# Main.@code_warntype jvp50!(symgraph, v0, xroots0, leaf0)
# @benchmark $jvp50!($symgraph, $v0, $xroots0, $leaf0)
num_allocs = @ballocated $jvp50!($symgraph, $v0, $xroots0, $leaf0)

jv1 = symgraph.jvp[1]
jv0 = FiniteDiff.finite_difference_gradient(x1 -> f1234(x1, xroots0[2])'*v0, xroots0[1])
norm(jv0, Inf)
norm(jv1, Inf)
norm(jv0 - jv1, Inf)


# Jacobian
jac0 = zeros(n+3,n)
jacobian50!(jac0, symgraph, xroots0, root0, leaf0)
# Main.@code_warntype jacobian50!(jac0, symgraph, xroots0, root0, leaf0)
# @benchmark $jacobian50!($jac0, $symgraph, $xroots0, $root0, $leaf0)
num_allocs = @ballocated $jacobian50!($jac0, $symgraph, $xroots0, $root0, $leaf0)

jac1 = FiniteDiff.finite_difference_jacobian(x1 -> f1234(x1, xroots0[2]), xroots0[1])
norm(jac0, Inf)
norm(jac1, Inf)
norm(jac0 - jac1, Inf)
