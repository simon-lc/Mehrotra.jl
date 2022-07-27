using Test

################################################################################
# test
################################################################################
function f1(x0::Vector)
    x1 = x0 .+ 1
end

function f2(x1::Vector)
    x2 = [x1; sum(x1); log(abs(sum(cos.(x1))))]
end

function f3(x2::Vector)
    x3 = x2.^2 * x2[end] * x2[end-1]
end

function f4(x3::Vector)
    x4 = [sum(x3); sin.(x3 * x3[end])]
end

function f1234(x0::Vector; output_layer::Int=-1, current_layer::Int=0, input_dim::Int=0)
    @layer x0
    x1 = x0 .+ 1
    @layer x1
    x2 = [x1; sum(x1); log(abs(sum(cos.(x1))))]
    @layer x2
    x3 = x2.^2 * x2[end] * x2[end-1]
    @layer x3
    x4 = [sum(x3); sin.(x3 * x3[end])]
    @layer x4
    x5 = x4
    @layer x5
    return x4
end

################################################################################
# example
################################################################################

n = 3
x0 = ones(n)
E0 = f1234(x0)
E1 = f4(f3(f2(f1(x0))))

J0 = FiniteDiff.finite_difference_jacobian(x -> f1234(x), x0)
J1 = FiniteDiff.finite_difference_jacobian(x -> f4(f3(f2(f1(x)))), x0)
@test norm(E0 - E1) < 1e-10
@test norm(J0 - J1) < 1e-10

symnet = generate_symnet(f1234, n)
monolith_evaluation!(symnet, x0)
monolith_jacobian!(symnet, x0)
@test norm(symnet.xouts[end] - E0) < 1e-5
@test norm(symnet.Couts[end] - J0) < 1e-5
# Main.@code_warntype monolith_evaluation!(symnet, x0)
@benchmark $monolith_evaluation!($symnet, $x0)
@benchmark $monolith_jacobian!($symnet, $x0)

evaluation!(symnet, x0)
jacobian!(symnet, x0)
@test norm(symnet.xouts[end] - E0) < 1e-5
@test norm(symnet.Couts[end] - J0) < 1e-5
# Main.@profiler [eval_fcts!(symnet, x0) for i=1:10000000]
# Main.@code_warntype evaluation!(symnet, x0)
@benchmark $evaluation!($symnet, $x0)
@benchmark $jacobian!($symnet, $x0)

# f1234(x0)
# f1234(x0, output_layer=-1)
# f1234(x0, output_layer=0)
# f1234(x0, output_layer=1)
# f1234(x0, output_layer=2)
# f1234(x0, output_layer=3)
# f1234(x0, output_layer=4)
# f1234(x0, output_layer=5)
# f1234(x0, output_layer=6)
# x0[1]
