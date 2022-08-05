using GLVisualizer
using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays

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
friction_coefficient = 0.1

mech = get_bundle_drop(;
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

################################################################################
# GLVisualizer point-cloud
################################################################################
resolution = (301, 700)
glvis = GLVisualizer.Visualizer(resolution=resolution)
open(glvis)
GLVisualizer.set_camera!(glvis;
    eyeposition=[0,+2,3.0],
    lookat=[0,0,0.0],
    up=[0,1,0.0])

build_mechanism!(glvis, mech)
GLVisualizer.set_floor!(glvis)

p1 = [Int(floor((resolution[1]+1)/2))] # perfectly centered for uneven number of pixels
p2 = Vector(1:20:resolution[2])
n1 = length(p1)
n2 = length(p2)

include("../cvxnet/softmax.jl")
include("../cvxnet/loss.jl")
include("../cvxnet/point_cloud.jl")

A0 = [
    +1.0 -0.2;
    +0.0 +1.0;
    -1.0 -0.2;
    +0.0 -1.0;
    ]
for i = 1:4
    A0[i,:] ./= norm(A0[i,:])
end
b0 = 0.5*[
    +1.5,
    +1,
    +1.5,
    +0,
    ];
A1 = [
    +1.0 +0.3;
    +0.0 +1.0;
    -1.0 +0.1;
    +0.0 -1.0;
    ]
for i = 1:4
    A1[i,:] ./= norm(A1[i,:])
end
b1 = 0.5*[
    +0.5,
    +2,
    +1,
    -1,
    ];
A2 = [
    +1.0 -0.3;
    +0.0 +1.0;
    -1.0 +0.5;
    +0.0 -1.0;
    ]
for i = 1:4
    A2[i,:] ./= norm(A2[i,:])
end
b2 = 0.5*[
    +0.75,
    +3,
    +0,
    -1,
    ];

# build_2d_polytope!(vis, A0, b0, name=:reference1,
#     color=RGBA(0.2,0.2,0.8,1.0))
# build_2d_polytope!(vis, A1, b1, name=:reference2,
#     color=RGBA(0.7,0.2,0.6,1.0))
# build_2d_polytope!(vis, A2, b2, name=:reference3,
#     color=RGBA(0.4,0.6,0.7,1.0))
# settransform!(vis[:reference1], MeshCat.Translation(+0.1, 0,0))
# settransform!(vis[:reference2], MeshCat.Translation(+0.2, 0,0))
# settransform!(vis[:reference3], MeshCat.Translation(+0.3, 0,0))



eyeposition = [0,0,3.0]
lookat = [0,0,0.5]
up = [0,1,0.0]
E = [3*[0, cos(α), sin(α)] for α in range(0.1π, 0.9π, 5)]
z_nominal = zeros(6)
set_mechanism!(vis, mech, z_nominal)
WC = [point_cloud(glvis, mech, z_nominal, p1, p2, resolution, e, lookat, up) for e in E]
for i = 1:5
    for j = 1:n1*n2
        setobject!(
            vis[:poincloud][Symbol(i)][Symbol(j)],
            HyperSphere(MeshCat.Point(0,0,0.0), 0.025),
            MeshPhongMaterial(color=RGBA(0.1,0.1,0.1,1)))
        settransform!(vis[:poincloud][Symbol(i)][Symbol(j)], MeshCat.Translation(WC[i][:,j]...))
    end
end

θ0, bundle_dimensions0 = pack_halfspaces([A0, A1], [b0, b1], [zeros(2), zeros(2)])
δ0 = 1e+2
Δ0 = 2e-2
βb0 = 3e-3
βo0 = 1e-3
βf0 = 3e-1
plt = plot_polytope(A0, b0, δ0)
plt = plot_polytope(A1, b1, δ0)
plt = plot_polytope(A2, b2, δ0)
bundle_dimensions = [4,4,4]
@time loss(WC, E, bundle_dimensions0, θ0, βb=βb0, βo=βo0, βf=βf0, Δ=Δ0, δ=δ0)

n = 5
bundle_dimensions = [n,n,n,n,n]
nb = length(bundle_dimensions)
local_loss(θ) = loss(WC, E, bundle_dimensions, θ, βb=βb0, βo=βo0, βf=βf0, Δ=Δ0, δ=δ0)

local_initial_invH(θ) = zeros(nb*(2n+2),nb*(2n+2)) + 1e-3*I

θinit = zeros(0)
for i = 1:nb
    θi = [range(-π, π, length=n+1)[1:end-1]; 0.40*ones(n); zeros(2)] + 0.5*rand(2n+2)
    A, b, o = unpack_halfspaces(θi)
    b .+= A * [0.2*(-1+2i/nb), 0.5]
    push!(θinit, pack_halfspaces(A, b, o)...)
end
θinit
Ainit, binit, oinit = unpack_halfspaces(θinit, bundle_dimensions)
local_loss(θinit)
@elapsed res = Optim.optimize(local_loss, θinit,
    # BFGS(),
    # Newton(),
    BFGS(initial_invH = local_initial_invH),
    autodiff=:forward,
    Optim.Options(
        # f_tol=1e-3,
        allow_f_increases=true,
        iterations=100,
        # callback=local_callback,
        extended_trace = true,
        store_trace = true,
        show_trace = false),
    )
# fieldnames(typeof(Optim.trace(res)[1])).iteration

################################################################################
# unpack results
################################################################################
θopt = Optim.minimizer(res)
Aopt, bopt, oopt = unpack_halfspaces(θopt, bundle_dimensions)
solution_trace = [iterate.metadata["x"] for iterate in Optim.trace(res)]
plot(hcat(solution_trace...)', legend=false)

################################################################################
# visualize results
################################################################################
build_2d_polytope!(vis[:reference], A0, b0, name=Symbol(1),
    color=RGBA(0.2,0.2,0.8,1.0))
build_2d_polytope!(vis[:reference], A1, b1, name=Symbol(2),
    color=RGBA(0.2,0.2,0.8,1.0))
build_2d_polytope!(vis[:reference], A2, b2, name=Symbol(3),
    color=RGBA(0.2,0.2,0.8,1.0))
for i = 1:nb
    # build_2d_polytope!(vis, Ainit[i], binit[i] + Ainit[i]*oinit[i], name=Symbol(:initial, i),
    build_2d_polytope!(vis[:initial], Ainit[i], binit[i], name=Symbol(i),
            color=RGBA(1,0.3,0.0,0.5))
end

for j = 1:length(solution_trace)
    for i = 1:nb
        Aopt, bopt, oopt = unpack_halfspaces(solution_trace[j], bundle_dimensions)
        try
            # build_2d_polytope!(vis[:optimized][Symbol(i)], Aopt[i], bopt[i] + Aopt[i]*oopt[i], name=Symbol(j),
            build_2d_polytope!(vis[:optimized][Symbol(i)], Aopt[i], bopt[i], name=Symbol(j),
                color=RGBA(1,1,0.0,0.5))
        catch e
        end
    end
end

anim = MeshCat.Animation(20)
for j = 1:length(solution_trace)
    atframe(anim, j) do
        for jj = 1:length(solution_trace)
            for i = 1:nb
                setvisible!(vis[:optimized][Symbol(i)][Symbol(jj)], j==jj)
            end
        end
    end
end
MeshCat.setanimation!(vis, anim)

settransform!(vis[:reference], MeshCat.Translation(+0.0, 0,0))
for i = 1:nb
    settransform!(vis[:initial][Symbol(i)], MeshCat.Translation(+0.1i, 0,0))
end
for i = 1:nb
    settransform!(vis[:optimized][Symbol(i)], MeshCat.Translation(+0.1(nb+i), 0,0))
end

# vis = Visualizer()
# open(vis)
set_floor!(vis, x=1.0, origin=[-0.5,0,0])
set_light!(vis, direction="Negative", ambient=0.7)
set_background!(vis)

RobotVisualizer.set_camera!(vis, zoom=40.0)

# RobotVisualizer.convert_frames_to_video_and_gif("learning_bundle_point_cloud")
