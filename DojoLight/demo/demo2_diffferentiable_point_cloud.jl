using GLVisualizer
using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays
using Clustering


vis = Visualizer()
open(vis)
set_floor!(vis, x=1.0, origin=[-0.5,0,0])
set_light!(vis, direction="Negative", ambient=0.7)
set_background!(vis)

include("../src/DojoLight.jl")

include("../cvxnet/softmax.jl")
include("../cvxnet/loss.jl")
include("../cvxnet/point_cloud.jl")

colors = [
    RGBA(1,0,0,1),
    RGBA(0,1,0,1),
    RGBA(0,0,1,1),
    RGBA(0,1,1,1),
    RGBA(1,1,0,1),
    RGBA(1,0,1,1),
    RGBA(0.5,0.5,0.5,1),
]

################################################################################
# ground-truth polytope
################################################################################
A0 = [
    +1.0 +0.0;
    +0.0 +1.0;
    -1.0 +0.4;
    +0.0 -1.0;
    ]
for i = 1:4
    A0[i,:] ./= norm(A0[i,:])
end
b0 = 0.5*[
    +1.5,
    +1.0,
    +1.5,
    +0,
    ];
o0 = [0, 0.0]

Af = [0 1.0]
bf = [0.0]
of = [0, 0.0]

θ_ref, bundle_dimensions_ref = pack_halfspaces([A0, Af], [b0, bf], [o0, of])

################################################################################
# ground-truth point-cloud
################################################################################
build_2d_polytope!(vis, A0, b0 + A0 * o0)

eyeposition = [0,0,3.0]
lookat = [0,0,0.0]
ne = 5
nβ = 20
e = [1.5*[cos(α), sin(α)] for α in range(0.2π, 0.8π, ne)]
β = [range(-0.2π, -0.8π, length=nβ) for i = 1:ne]

δ = 10.0
P = [sumeet_point_cloud(e[i], β[i], θ_ref, bundle_dimensions_ref, δ) for i = 1:ne]

build_2d_point_cloud!(vis, P, e)

plt = plot()
for i = 1:ne
    plot!(plt, [norm(P[i][:,j]) for j=1:nβ])
end
display(plt)


################################################################################
# initialization
################################################################################
# point cloud reaching the object
Pobject = []
for i = 1:ne
    for j = 1:nβ
        p = P[i][:,j]
        if p[2] > 1e-3
            push!(Pobject, p)
        end
    end
end
Pobject = hcat(Pobject...)

# convex bundle parameterization
nh = 5
bundle_dimensions = [nh, nh, nh]
np = length(bundle_dimensions)

# k-mean clustering
kmres = kmeans(Pobject, np)
# display k-mean result
for i = 1:size(Pobject, 2)
    ik = kmres.assignments[i]
    setobject!(
        vis[:cluster][Symbol(ik)][Symbol(i)],
        HyperSphere(MeshCat.Point(0,0,0.0), 0.035),
        MeshPhongMaterial(color=colors[ik]))
    settransform!(vis[:cluster][Symbol(ik)][Symbol(i)], MeshCat.Translation(0.1, Pobject[:,i]...))
end
for i = 1:np
    setobject!(
        vis[:cluster][Symbol(i)][:center],
        HyperRectangle(MeshCat.Vec(-0.05,-0.05,-0.05), MeshCat.Vec(0.1,0.1,0.1)),
        MeshPhongMaterial(color=colors[i]))
    settransform!(vis[:cluster][Symbol(i)][:center], MeshCat.Translation(0.2, kmres.centers[:,i]...))
end

# initialization
θinit = zeros(0)
for i = 1:np
    θi = [range(-π, π, length=nh+1)[1:end-1]; 0.25*ones(nh); kmres.centers[:, i]]
    A, b, o = unpack_halfspaces(θi)
    push!(θinit, pack_halfspaces(A, b, o)...)
end
build_2d_convex_bundle!(vis, θinit, bundle_dimensions, name=:initial, color=RGBA(1,0,0,0.4))

################################################################################
# optimization
################################################################################
# optimization parameters
δsoft = 10.0
δ0 = 1e+2
Δ0 = 2e-2
βb0 = 3e-3
βo0 = 1e-3
βf0 = 3e-1


# loss
local_loss(θ) = sumeet_loss(P, e, β, θ, bundle_dimensions, δsoft)
local_initial_invH(θ) = zeros(np * (2nh+2), np * (2nh+2)) + 1e-1*I

# initialization
sumeet_loss(P, e, β, θinit, bundle_dimensions, δsoft)
@time local_loss(θinit)

@elapsed res = Optim.optimize(local_loss, θinit,
    # BFGS(initial_invH = local_initial_invH),
    BFGS(),
    # Newton(),
    autodiff=:forward,
    Optim.Options(
        # allow_f_increases=true,
        iterations=2000,
        # callback=local_callback,
        extended_trace = true,
        store_trace = true,
        # show_trace = false,
        show_trace = true,
    ))

################################################################################
# unpack results
################################################################################
θopt = Optim.minimizer(res)
# solution_trace = [iterate.metadata["centroid"] for iterate in Optim.trace(res)[1:40:end]]
# solution_trace = [iterate.metadata["centroid"] for iterate in Optim.trace(res)]
solution_trace = [iterate.metadata["x"] for iterate in Optim.trace(res)]
plot(hcat(solution_trace...)', legend=false)

plot(θinit)
plot(θopt)
plot(θinit - θopt)
scatter(θinit - θopt)

################################################################################
# visualize results
################################################################################
for i = 1:length(solution_trace)
    try
        build_2d_convex_bundle!(vis[:trace], solution_trace[i], bundle_dimensions, name=Symbol(i), color=RGBA(1,1,0,0.4))
    catch e
    end
end

anim = MeshCat.Animation(20)
for i = 1:length(solution_trace)
    atframe(anim, i) do
        for ii = 1:length(solution_trace)
            for j = 1:np
                # @show i
                # @show ii
                @show i == ii
                setvisible!(vis[:trace][Symbol(ii)][Symbol(j)], i == ii)
            end
        end
    end
end
MeshCat.setanimation!(vis, anim)

settransform!(vis[:trace], MeshCat.Translation(+0.2, 0,0))

RobotVisualizer.set_camera!(vis, zoom=40.0)
