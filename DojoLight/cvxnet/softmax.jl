################################################################################
# softmax
################################################################################
function sdf(p, A, b, δ)
    N = length(b)
    v = δ[1] * (A*p - b)
    vm = maximum(v)
    e = exp.(v .- vm)
    s = 1/N * sum(e) # average
    ϕ = (log(s) + vm) / δ[1]
    return ϕ
end

function squared_sdf(p, A, b, δ)
    N = length(b)
    v = δ[1] * (A*p - b)
    vm = maximum(v)
    e = exp.(v .- vm)
    s = 1/N * sum(e) # average
    ϕ = (log(s) + vm) / δ[1]
    return [ϕ]
end

function gradient_sdf(p, A, b, δ)
    N = length(b)
    v = δ[1] * (A*p - b)
    vm = maximum(v)
    e = exp.(v .- vm)
    s = 1/N * sum(e) # average
    ϕ = log(s) / δ[1]

    g = 1/(s * N) * A' * e
    return g
end

function hessian_sdf(p, A, b, δ)
    N = length(b)
    v = δ[1] * (A*p - b)
    vm = maximum(v)
    e = exp.(v .- vm)
    s = 1/N * sum(e) # average
    ϕ = log(s) / δ[1]


    dedx = δ * Diagonal(e) * A
    dsdx = 1/N * δ * e' * A
    H = 1/(s * N) * A' * dedx
    H += 1/N * A' * e * (-1/s^2) * dsdx
    return H
end


function plot_halfspace(plt, a, b; show_plot::Bool=true)
    R = [0 1; -1 0]
    v = R * a[1,:]
    x0 = a \ b
    xl = x0 - 100*v
    xu = x0 + 100*v
    plt = plot!(plt, [xl[1], xu[1]], [xl[2], xu[2]], linewidth=5.0, color=:white)
    show_plot && display(plt)
    return plt
end

function plot_polytope(A, b, δ;
        xlims=(-1,1), ylims=(-1,1), S::Int=25) where {T,N,D}

    X = range(xlims..., length=S)
    Y = range(ylims..., length=S)
    V = zeros(S,S)

    for i = 1:S
        for j = 1:S
            p = [X[i], Y[j]]
            V[j,i] = sdf(p, A, b, δ)[1]
        end
    end

    plt = heatmap(
        X, Y, V,
        aspectratio=1.0,
        xlims=xlims,
        ylims=ylims,
        xlabel="x", ylabel="y",
        )
    for i = 1:length(b)
        plt = plot_halfspace(plt, A[i:i,:], b[i:i], show_plot=false)
    end
    plt = contour(plt, X,Y,V, levels=[0.0], color=:black, linewidth=2.0)
    return plt
end

#
# A = [
#     +1.0 +0.0;
#     +0.0 +1.0;
#     +0.0 -1.0;
#     -1.0  0.0;
#     ]
# b = 0.5*[
#     1,
#     1,
#     1,
#     1.,
#     ]
# δ = 1e2
#
# x = 100*[2,1.0]
# ϕ0 = sdf(x, A, b, δ)
# g0 = FiniteDiff.finite_difference_gradient(x -> sdf(x, A, b, δ), x)
# H0 = FiniteDiff.finite_difference_hessian(x -> sdf(x, A, b, δ), x)
#
# g1 = gradient_sdf(x, A, b, δ)
# H1 = hessian_sdf(x, A, b, δ)
# norm(g0 - g1, Inf)
# norm(H0 - H1, Inf)
#
# δ = 1e2
# plot_polyhedron(A, b, δ)


function softmax_mean(values, weight, δ)
    n = length(values)

    wmax = maximum(weight)
    normalizer = 0.0
    for i = 1:n
        normalizer += exp(δ * (weight[i] - wmax))
    end

    m = 0.0
    for i = 1:n
        m += values[i] * exp(δ * (weight[i] - wmax)) / normalizer
    end
    return m
end

function softmax(values, δ)
    n = length(values)

    vmax = maximum(values)
    normalizer = 0.0
    for i = 1:n
        normalizer += exp(δ * (values[i] - vmax))
    end
    m = 0.0
    for v in values
        (v == -Inf) && continue
        m += v * exp(δ * (v - vmax)) / normalizer
    end
    return m
end

function softweights(values::Vector{T}, δ) where T
    n = length(values)
    w = zeros(T, n)

    vmax = maximum(values)
    normalizer = 0.0
    for i = 1:n
        normalizer += exp(δ * (values[i] - vmax))
    end

    for (i,v) in enumerate(values)
        (v == -Inf) && continue
        w[i] = exp(δ * (v - vmax)) / normalizer
    end
    return w
end

function softmin(values, δ)
    return -softmax(-values, δ)
end
softmin([-1, 2, 3, 4, -1, 3], 3)

sum(softweights(rand(10), 2.0))