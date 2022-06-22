using LinearAlgebra
using Plots

mutable struct Polyhedron113{T,N,D}
    A::Matrix{T}
    b::Vector{T}
    n::Int
    d::Int
end

function Polyhedron113(A::Matrix{T}, b::Vector{T}) where T
    n, d = size(A)
    @assert length(b) == n
    for i = 1:n
        A[i,:] .= normalize(A[i,:])
    end
    return Polyhedron113{T,n,d}(A, b, n, d)
end

function sdf(p::Vector{T}, poly::Polyhedron113{T,N,D}) where {T,N,D}
    A = poly.A
    b = poly.b
    ϕ = maximum(A*p - b)
    return [ϕ]
end

function residual(primals, duals, slacks, parameters)
    A1 = poly_1.A
    A2 = poly_2.A
    b1 = poly_1.b
    b2 = poly_2.b
    n1 = poly_1.n
    n2 = poly_2.n

    y, z, s = primals, duals, slacks
    y1 = y[1:3]
    y2 = y[4:6]
    z1 = z[1:n1]
    z2 = z[n1 .+ (1:n2)]
    s1 = s[1:n1]
    s2 = s[n1 .+ (1:n2)]

    res = [
        (y1 - y2) - A1' * z1;
        (y2 - y1) - A2' * z2;
        s1 - (A1 * y1 - b1);
        s2 - (A2 * y2 - b2);
        # s1 .* γ1;
        # s2 .* γ2;
    ]
    return res
end

A1 = [
    +1.0 +0.0 +0.0;
    +0.0 +1.0 +0.0;
    -1.0  0.0 +0.0;
    +0.0 -1.0 +0.0;
    ]
b1 = 0.5*[
    +3,
    +1,
    -1,
    +1,
    ]

A2 = [
    +1.0 +0.0 +0.0;
    +0.0 +1.0 +0.0;
    -1.0  0.0 +0.0;
    +0.0 -1.0 +0.0;
    ]
b2 = 0.5*[
    +1,
    +1,
    +1,
    +1.,
    ]


poly_1 = Polyhedron113(A1, b1)
poly_2 = Polyhedron113(A2, b2)

p_poly = [0, 0.0]
θ_poly = [0.0]
# plt = plot_polyhedron(p_poly, θ_poly, poly_1, xlims=(-3,3))
# plt = plot_polyhedron(p_poly, θ_poly, poly_2, xlims=(-3,3))



num_primals = 6
num_cone = poly_1.n + poly_2.n
num_parameters = 1
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]
parameters = zeros(1)


solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options228(max_iterations=30, verbose=true)
    )


solve!(solver)
solver.solution
J = Matrix(solver.data.jacobian_variables)
plot(Gray.(abs.(J)))
norm(J - J')


function contact_point(p::Vector{T}, θ::Vector{T}, poly::Polyhedron113{T,N,D}) where {T,N,D}
    # q = [x, y, θ]
    R = [cos(θ[1]) sin(θ[1]);
        -sin(θ[1]) cos(θ[1])]
    c = [0.0]
    for k = 1:5
        @show k
        for i = 1:100
            e = sdf(R * ([c; 0.0] - p), poly)
            # @show e
            r = FiniteDiff.finite_difference_gradient(c -> sdf(R * ([c; 0.0] - p), poly)[1], c)
            # @show r
            @show norm(r, Inf)
            (norm(r, Inf) < 1e-10) && break
            H = FiniteDiff.finite_difference_hessian(c -> sdf(R * ([c; 0.0] - p), poly)[1], c)
            # @show H
            H[1,1] = max(H[1,1], 1e-6)
            # @show H
            Δ = -H \ r
            # @show Δ
            α = 1.0
            for i = 1:100
                e_cand = sdf(R * ([c + α * Δ; 0.0] - p), poly)
                (e_cand[1] <= e[1]) && break
                α /= 2
            end
            # @show α
            c = c + α * Δ
            # @show c
        end
        poly.δ *= sqrt(10.0)
    end
    return [c; 0.0]
end

function plot_halfspace(plt, a, b)
    R = [0 1; -1 0]
    v = R * a[1,:]
    x0 = a \ b
    xl = x0 - 100*v
    xu = x0 + 100*v
    plt = plot!(plt, [xl[1], xu[1]], [xl[2], xu[2]], linewidth=5.0, color=:white)
    display(plt)
    return plt
end

function plot_polyhedron(p, θ, poly::Polyhedron113{T,N,D};
        xlims=(-1,1), ylims=(-1,1), S::Int=100) where {T,N,D}

    X = range(xlims..., length=S)
    Y = range(ylims..., length=S)
    V = zeros(S,S)

    for i = 1:S
        for j = 1:S
            p = [X[i], Y[j]]
            V[j,i] = sdf(p, poly)[1]
        end
    end

    plt = heatmap(
        X, Y, V,
        aspectratio=1.0,
        xlims=xlims,
        ylims=ylims,
        c=cgrad([:blue, :white,:red, :yellow]),
        xlabel="x", ylabel="y",
        )
    for i = 1:N
        plt = plot_halfspace(plt, poly.A[i:i,:], poly.b[i:i])
    end
    plt = contour(plt, X,Y,V, levels=[0.0], color=:black)
    return plt
end


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

poly = Polyhedron113(A, b, δ=1e+10)
p = [1,110.0]
sdf(p, poly)
p_poly = [200, 0.001]
θ_poly = [0.05]
plt = plot_polyhedron(p_poly, θ_poly, poly)

contact_point(p_poly, θ_poly, poly)[1]


e = sdf([0.0; 0.0] - p_poly, poly)
r = FiniteDiff.finite_difference_gradient(c -> sdf([c; 0.0] - p_poly, poly)[1], [0.0])


# A = [
#     +1.0 +0.0;
#     +1.0 +1.0;
#     +0.4 -1.0;
#     -1.0  0.4;
#     ]
# b = [0.75,
#     1,
#     1,
#     0.5
#     ]
#
