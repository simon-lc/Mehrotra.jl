using GeometryBasics
using GLVisualizer
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim


vis = Visualizer()
open(vis)

include("../src/DojoLight.jl")

################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

mech = get_sphere_bundle(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        # verbose=true,
        verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        # complementarity_decoupling=true
        )
    );

# Main.@profiler solve!(mech.solver)
################################################################################
# test simulation
################################################################################
xp2 = [+0.1,1.5,-0.25]
xc2 = [-0.0,0.5,-2.25]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]

u0 = zeros(6)
H0 = 100

@elapsed storage = simulate!(mech, z0, H0)
# Main.@profiler [solve!(mech.solver) for i=1:300]
# @benchmark $solve!($(mech.solver))

# scatter(storage.iterations)
# plot!(hcat(storage.variables...)')

################################################################################
# MeshCat visualization
################################################################################
set_floor!(vis)
set_light!(vis)
set_background!(vis)
vis, anim = visualize!(vis, mech, storage, build=true)
#
# ################################################################################
# # GLVisualizer visualization
# ################################################################################
# glvis = GLVisualizer.Visualizer()
# open(glvis)
#
# GLVisualizer.set_camera!(glvis;
#     eyeposition=[5,0,0.5],
#     lookat=[0,0,0.0],
#     up=[0,0,1.0])
# GLVisualizer.set_floor!(glvis)
#
# build_mechanism!(glvis, mech)
# for i = 1:H0
#     # sleep(timestep)
#     set_mechanism!(glvis, mech, storage, i)
# end


################################################################################
# GLVisualizer point-cloud
################################################################################
resolution = (300, 700)
glvis = GLVisualizer.Visualizer(resolution=resolution)
open(glvis)
GLVisualizer.set_camera!(glvis;
    eyeposition=[0,-4,3.0],
    lookat=[0,0,0.0],
    up=[0,1,0.0])

build_mechanism!(glvis, mech)
GLVisualizer.set_floor!(glvis)

p1 = [Int(floor(resolution[1]/2))]
p2 = Vector(1:10:resolution[2])
n1 = length(p1)
n2 = length(p2)

depth = zeros(Float32, resolution...)
world_coordinates = zeros(3, n1 * n2)
pixel = HyperSphere(GeometryBasics.Point(0,0,0.0), 0.025)
for j = 1:n1*n2
    setobject!(vis[:pointcloud][Symbol(j)], pixel, MeshPhongMaterial(color=RGBA(0,0,0,1)))
end
for i = 1:H0
    atframe(anim, i) do
        set_mechanism!(glvis, mech, storage, i)
        depth_buffer!(depth, glvis)
        depthpixel_to_world!(world_coordinates, depth, p1, p2, glvis)
        for j in 1:n1*n2
            settransform!(vis[:pointcloud][Symbol(j)], MeshCat.Translation(world_coordinates[:,j]))
        end
    end
end
MeshCat.setanimation!(vis, anim)

linear_depth = (depth .- minimum(depth)) ./ (1e-5+(maximum(depth) - minimum(depth)))
plot(Gray.(linear_depth))


# RobotVisualizer.convert_frames_to_video_and_gif("point_cloud_sphere_bundle_tilt")


function point_cloud(vis::GLVisualizer, A::Vector{Matrix}, b::Vector{Vector}, eyeposition, lookat, up)

    GLVisualizer.set_camera!(glvis;
        eyeposition=eyepositionl,
        lookat=lookat,
        up=up)



    return world_coordinates
end



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

function plot_polyhedron(A, b, δ;
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
        plt = plot_halfspace(plt, A[i:i,:], b[i:i])
    end
    plt = contour(plt, X,Y,V, levels=[0.0], color=:black, linewidth=2.0)
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
δ = 1e2

x = 100*[2,1.0]
ϕ0 = sdf(x, A, b, δ)
g0 = FiniteDiff.finite_difference_gradient(x -> sdf(x, A, b, δ), x)
H0 = FiniteDiff.finite_difference_hessian(x -> sdf(x, A, b, δ), x)

g1 = gradient_sdf(x, A, b, δ)
H1 = hessian_sdf(x, A, b, δ)
norm(g0 - g1, Inf)
norm(H0 - H1, Inf)


x = range(-2, +2, length = 25)
y = range(-2, +2, length = 25)
ϕ = [sdf([xi, yi], A, b, δ) for xi in x, yi in y]
plt = heatmap(plt, x, y, ϕ, aspectratio=1.0)
plt = contour(plt, x, y, ϕ, levels=[0.0], color=:white)



δ = 1e2
plot_polyhedron(A, b, δ)
