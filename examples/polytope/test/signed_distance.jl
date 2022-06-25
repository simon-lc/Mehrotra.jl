################################################################################
# test
################################################################################
A = [
    +1.0 +0.0;
    +0.0 +1.0;
    +0.0 -1.0;
    -1.0  0.0;
    ]
b = 0.5*[
    1,
    1,
    1,
    1.,
    ]
δ = 1e-3
x = 100*[2,1.0]


ϕ0 = squared_signed_distance(x, A, b, δ)
g0 = FiniteDiff.finite_difference_gradient(x -> squared_signed_distance(x, A, b, δ), x)
H0 = FiniteDiff.finite_difference_hessian(x -> squared_signed_distance(x, A, b, δ), x)
g1 = ForwardDiff.gradient(x -> squared_signed_distance(x, A, b, δ), x)
H1 = ForwardDiff.hessian(x -> squared_signed_distance(x, A, b, δ), x)

g2 = gradient_squared_signed_distance(x, A, b, δ)
H2 = hessian_squared_signed_distance(x, A, b, δ)
norm(g0 - g2, Inf)
norm(H0 - H2, Inf)
norm(g1 - g2, Inf)
norm(H1 - H2, Inf)


ϕ0 = signed_distance(x, A, b, δ)
g0 = FiniteDiff.finite_difference_gradient(x -> signed_distance(x, A, b, δ), x)
H0 = FiniteDiff.finite_difference_hessian(x -> signed_distance(x, A, b, δ), x)
g1 = ForwardDiff.gradient(x -> signed_distance(x, A, b, δ), x)
H1 = ForwardDiff.hessian(x -> signed_distance(x, A, b, δ), x)

g2 = gradient_signed_distance(x, A, b, δ)
H2 = hessian_signed_distance(x, A, b, δ)
norm(g0 - g2, Inf)
norm(H0 - H2, Inf)
norm(g1 - g2, Inf)
norm(H1 - H2, Inf)
