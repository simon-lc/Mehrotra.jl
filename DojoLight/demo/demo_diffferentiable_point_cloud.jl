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
using ForwardDiff
using BenchmarkTools

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
    -1.0 +0.0;
    +0.0 -1.0;
    ] + 0.2*ones(4,2)
for i = 1:4
    A0[i,:] ./= norm(A0[i,:])
end
b0 = 0.5*[
    +1.0,
    +1.0,
    +0.5,
    +0,
    ] - 0.0*ones(4);
o0 = [0, 0.0]

A1 = [
    +1.0 +0.0;
    +0.0 +1.0;
    -1.0 +0.0;
    +0.0 -1.0;
    ] - 0.2*ones(4,2)
for i = 1:4
    A1[i,:] ./= norm(A1[i,:])
end
b1 = 0.5*[
    +1.0,
    +1.0,
    +0.5,
    +0,
    ] - 0.2*ones(4);
o1 = [-0.4, 0.0]

A2 = [
    +1.0 +0.0;
    +0.0 +1.0;
    -1.0 +0.0;
    +0.0 -1.0;
    ] + 0.2*ones(4,2)
for i = 1:4
    A2[i,:] ./= norm(A2[i,:])
end
b2 = 0.5*[
    +1.0,
    +1.0,
    +0.5,
    +0,
    ] - 0.1*ones(4);
o2 = [-0.2, +0.4]

Af = [0 1.0]
bf = [0.0]
of = [0, 0.0]

θ_ref, bundle_dimensions_ref = pack_halfspaces([A0, Af], [b0, bf], [o0, of])
θ_ref, bundle_dimensions_ref = pack_halfspaces([A0, A1, Af], [b0, b1, bf], [o0, o1, of])
θ_ref, bundle_dimensions_ref = pack_halfspaces([A0, A1, A2, Af], [b0, b1, b2, bf], [o0, o1, o2, of])

################################################################################
# ground-truth point-cloud
################################################################################
build_2d_polytope!(vis, A0, b0 + A0 * o0, name=:polytope_0)
build_2d_polytope!(vis, A1, b1 + A1 * o1, name=:polytope_1)
build_2d_polytope!(vis, A2, b2 + A2 * o2, name=:polytope_2)

eyeposition = [0,0,3.0]
lookat = [0,0,0.0]
ne = 2
nβ = 50
# e = [2.0*[cos(α), sin(α)] for α in range(0.5π, 0.5π, length=ne)]
e = [1.25*[cos(α), sin(α)] for α in range(0.4π, 0.6π, length=ne)]
β = [range(-0.2π, -0.8π, length=nβ) for i = 1:ne]

for δ = 6:1:20
    P = [sumeet_point_cloud(e[i], β[i], θ_ref, bundle_dimensions_ref, δ) for i = 1:ne]
    build_2d_point_cloud!(vis, P, e, name=:point_cloud)
    sleep(0.2)
end
δ = 100.0
P = [sumeet_point_cloud(e[i], β[i], θ_ref, bundle_dimensions_ref, δ) for i = 1:ne]
build_2d_point_cloud!(vis, P, e, name=:point_cloud)

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
bundle_dimensions = [nh, nh, nh, nh, nh]
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
    settransform!(vis[:cluster][Symbol(ik)][Symbol(i)], MeshCat.Translation(0.2, Pobject[:,i]...))
end
for i = 1:np
    setobject!(
        vis[:cluster][Symbol(i)][:center],
        HyperRectangle(MeshCat.Vec(-0.05,-0.05,-0.05), MeshCat.Vec(0.1,0.1,0.1)),
        MeshPhongMaterial(color=colors[i]))
    settransform!(vis[:cluster][Symbol(i)][:center], MeshCat.Translation(0.2, kmres.centers[:,i]...))
end
# initialization
b_ref = 2 * mean(sqrt.(kmres.costs))
θinit = zeros(0)
for i = 1:np
    angles = range(-π, π, length=nh+1)[1:end-1] + 0.15*rand(nh)
    # θi = [range(-π, π, length=nh+1)[1:end-1] + 0.15*rand(nh); b_ref*ones(nh); kmres.centers[:, i]]
    θi = [vcat([[cos(a), sin(a)] for a in angles]...); b_ref*ones(nh); kmres.centers[:, i]]
    A, b, o = unpack_halfspaces(θi)
    push!(θinit, pack_halfspaces(A, b, o)...)
end
θinit
build_2d_convex_bundle!(vis, θinit, bundle_dimensions, name=:initial, color=RGBA(1,1,0,0.4))


θdiag = zeros(0)
for i = 1:np
    # θi = [3e-2 * ones(nh); 1e-0 * ones(nh); 1e-0 * ones(2)]
    θi = [1e-0 * ones(2nh); 1e+1 * ones(nh); 1e+1 * ones(2)]
    A, b, o = unpack_halfspaces(θi)
    push!(θdiag, pack_halfspaces(A, b, o)...)
end
θdiag

################################################################################
# optimization
################################################################################
# optimization parameters
function add_floor(θ, bundle_dimensions)
    A, b, o = unpack_halfspaces(θ, bundle_dimensions)
    Af = [0 1.0]
    bf = [0.0]
    of = [0, 0.0]
    return pack_halfspaces([A..., Af], [b..., bf], [o..., of])
end
# loss
function local_loss(θ, δsoft; ω_centroid=1e-2, ω_offset=2e-3)
    θ_floor, bundle_dimensions_floor = add_floor(θ, bundle_dimensions)
    l = sumeet_loss(P, e, β, δsoft, θ_floor, bundle_dimensions_floor)

    A, b, o = unpack_halfspaces(θ, bundle_dimensions)
    np = length(bundle_dimensions)
    for i = 1:np
        l += 0.5 * (o[i] - kmres.centers[:,i])' * (o[i] - kmres.centers[:,i]) * ω_centroid
        # l += 0.5 * (b[i] .- b_ref)' * (b[i] .- b_ref) * ω_offset
        l += 0.5 * (b[i] .- 2b_ref)' * (b[i] .- 2b_ref) * ω_offset
    end
    return l
end


# initialization
sumeet_loss(P, e, β, δsoft, θinit, bundle_dimensions)
@time local_loss(θinit, δsoft)

# Glocal_loss(θ, δsoft) = ForwardDiff.gradient(θ -> local_loss(θ, δsoft), θ)
# Hlocal_loss(θ, δsoft) = ForwardDiff.hessian(θ -> local_loss(θ, δsoft), θ)
function block_diagonal_hessian(f::Function, blocks::Vector{Int})
    N = sum(blocks)
    nb = length(blocks)

    function block_f(θ, θi::Vector{T}, ind, params) where T
        θmixed = zeros(T, N)
        θmixed .= θ
        θmixed[ind] .= θi
        return f(θmixed, params)
    end

    function hessian(θ, params)
        H = spzeros(N, N)
        off = 0
        for i = 1:nb
            ind = off .+ (1:nb); off += nb
            H[ind, ind] .= ForwardDiff.hessian(θi -> block_f(θ, θi, ind, params), θ[ind])
        end
        return H
    end
    return hessian
end
nθ = 3 .* bundle_dimensions .+ 2
Hblock_loss = block_diagonal_hessian(local_loss, nθ)
Hb = Hblock_loss(θinit, δsoft)

# Ginit = Glocal_loss(θinit, δsoft)
# Hinit = Hlocal_loss(θinit, δsoft)
# plot(Gray.(abs.(Hinit)))

################################################################################
# projection
################################################################################
# θmin = vcat([[-Inf * ones(nh); +0.05 * ones(nh); -Inf * ones(2)] for i=1:np]...)
θmin = vcat([[-Inf * ones(2nh); +0.05 * ones(nh); -Inf * ones(2)] for i=1:np]...)
# θmax = vcat([[+Inf * ones(nh); +Inf  * ones(nh); +Inf * ones(2)] for i=1:np]...)
θmax = vcat([[+Inf * ones(2nh); +Inf  * ones(nh); +Inf * ones(2)] for i=1:np]...)
function projection(θ, θmin, θmax, bundle_dimensions)
    A, b, o = unpack_halfspaces(θ, bundle_dimensions)
    np = length(bundle_dimensions)
    for i = 1:np
        for j = 1:size(A[i],1)
            A[i][j,:] = A[i][j,:] / (1e-6 + norm(A[i][j,:]))
        end
    end
    return clamp.(θ, θmin, θmax)
end
local_projection(θ) = projection(θ, θmin, θmax, bundle_dimensions)

################################################################################
# clamping
################################################################################
Δθmin = vcat([[-0.15 * ones(2nh); -0.025 * ones(nh); -0.025 * ones(2)] for i=1:np]...)
Δθmax = vcat([[+0.15 * ones(2nh); +0.025 * ones(nh); +0.025 * ones(2)] for i=1:np]...)
function clamping(Δθ, Δθmin, Δθmax, bundle_dimensions)
    return clamp.(Δθ, Δθmin, Δθmax)
end
local_clamping(Δθ) = clamping(Δθ, Δθmin, Δθmax, bundle_dimensions)



################################################################################
# solve
################################################################################
θopt, solution_trace = mysolve!(θinit, local_loss, Glocal_loss,
    # Hlocal_loss, local_projection, max_iterations=100)
    Hblock_loss, local_projection, local_clamping, nθ, max_iterations=100)
# Main.@profiler θopt, solution_trace = mysolve!(θinit, local_loss, Glocal_loss,
    # Hlocal_loss, local_projection, max_iterations=30)
solution_trace = [solution_trace; fill(solution_trace[end], 20)]

θopt_floor, bundle_dimensions_floor = add_floor(θopt, bundle_dimensions)
Popt = [sumeet_point_cloud(e[i], β[i], θopt_floor, bundle_dimensions_floor, δ) for i = 1:ne]
build_2d_point_cloud!(vis, Popt, e, color=RGBA(0,0,1,0.5), name=:opt)
settransform!(vis[:opt], MeshCat.Translation(0.1, 0,0))
################################################################################
# unpack results
################################################################################
# θopt = Optim.minimizer(res)
plot(hcat(solution_trace...)', legend=false)

plt = plot(1 ./ abs.(θinit))
# plot!(plt, 1 ./ abs.(θdiag))
plot(θopt)
scatter(θinit - θopt)

################################################################################
# visualize results
################################################################################
for i = 1:length(solution_trace)
    try
        build_2d_convex_bundle!(vis[:trace], solution_trace[i], bundle_dimensions, name=Symbol(i), color=RGBA(1,0,0,0.4))
    catch error
    end
end

anim = MeshCat.Animation(20)
for i = 1:length(solution_trace)
    atframe(anim, i) do
        for ii = 1:length(solution_trace)
            for j = 1:np
                setvisible!(vis[:trace][Symbol(ii)][Symbol(j)], i == ii)
            end
        end
    end
end
MeshCat.setanimation!(vis, anim)

settransform!(vis[:trace], MeshCat.Translation(+0.4, 0,0))

RobotVisualizer.set_camera!(vis, zoom=10.0)
# RobotVisualizer.convert_frames_to_video_and_gif("complex_shape_convex_bundle")
