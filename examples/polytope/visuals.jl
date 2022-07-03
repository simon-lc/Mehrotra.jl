function build_polyhedron!(vis::Visualizer, A::Matrix{T}, b::Vector{T};
        name::Symbol=:polyhedron,
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    h = hrep(A, b)
    p = polyhedron(h)
    m = Polyhedra.Mesh(p)
    setobject!(vis[name], m, MeshPhongMaterial(color=color))
    return nothing
end

function build_2d_polyhedron!(vis::Visualizer, A::Matrix{T}, b::Vector{T};
        name::Symbol=:polyhedron,
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    n = size(A)[1]
    Ae = [zeros(n) A]
    Ae = [Ae;
         -1 0 0;
          1 0 0]
    be = [b; 0.1; 00]
    build_polyhedron!(vis, Ae, be, name=name, color=color)
    return nothing
end

# function set_polyhedron!(vis::Visualizer, p::Vector{T}, q::Quaternion{T};
function set_polyhedron!(vis::Visualizer, p::Vector{T}, q::Vector{T};
        name::Symbol=:polyhedron) where T

    settransform!(vis[name], MeshCat.compose(
        MeshCat.Translation(p...),
        MeshCat.LinearMap(z_rotation(q)),
        )
    )
    return nothing
end

function set_2d_polyhedron!(vis::Visualizer, p::Vector{T}, q::Vector{T};
        name::Symbol=:polyhedron) where T
    pe = [0; p]

    settransform!(vis[name], MeshCat.compose(
        MeshCat.Translation(SVector{3}(pe)),
        MeshCat.LinearMap(rotationmatrix(RotX(q[1]))),
        )
    )
    return nothing
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

function plot_polyhedron(p, θ, poly::Polyhedron{T,N,D};
        xlims=(-1,1), ylims=(-1,1), S::Int=100) where {T,N,D}

    X = range(xlims..., length=S)
    Y = range(ylims..., length=S)
    V = zeros(S,S)

    for i = 1:S
        for j = 1:S
            p = [X[i], Y[j]]
            V[j,i] = signed_distance(p, poly)[1]
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
