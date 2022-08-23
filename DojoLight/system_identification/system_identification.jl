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
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

include("../src/DojoLight.jl")

include("../cvxnet/softmax.jl")
include("../cvxnet/point_cloud.jl")
include("../cvxnet/halfspace.jl")
include("../cvxnet/visuals.jl")

include("../system_identification/newton_solver.jl")
include("../system_identification/structure.jl")
include("../system_identification/visuals.jl")


################################################################################
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
################################################################################
# partial observability: we currently observe the whole state
# works with multiple bodies and especially finger that are of known geometry
# need to integrate sphere to the point cloud generator
# speedup
# gradient and hessian approximated
# generic optimizer
# work with arbitrary number of shapes and bodies



################################################################################
# Cvx instantiation of System Identification
################################################################################

## measurement
struct CvxMeasurement1310{T} <: Measurement{T}
    # x::Vector{T} # polytope position
    # q::Vector{T} # polytope orientation
    z::Vector{T} # polytope state, we assume the whole state is observed TODO
    d::Vector{Matrix{T}} # point cloud obtained from several cameras
end

function unpack(θ::Vector, polytope_dimensions::Vector{Int})
    off = 0
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ (1:1)]; off += 1
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    A, b, o = unpack_halfspaces(θ[off + 1:end], polytope_dimensions)
    return mass, inertia, friction_coefficient, A, b, o
end

# camera
struct Camera1310{T}
    eye_position::Vector{T}
    camera_rays
    look_at::Vector{T}
    softness::T
end

## context
struct CvxContext1310{T} <: Context{T}
    mechanism::Mechanism1170
    cameras::Vector{Camera1310{T}}
    polytope_dimensions::Vector{Int}
end

function CvxContext1310(mechanism::Mechanism1170, cameras::Vector{Camera1310{T}}) where T
    z = zeros(mechanism.dimensions.state)
    A, b, o = get_halfspaces(mechanism, z)
    θ_geometry, polytope_dimensions = pack_halfspaces(A, b, o)
    return CvxContext1310(mechanism, cameras, polytope_dimensions)
end

function unpack(θ, context::CvxContext1310)
    polytope_dimensions = context.polytope_dimensions
    return unpack(θ, polytope_dimensions)
end

## objective
struct CvxObjective1310{T} <: Objective{T}
    P_state::Diagonal{T, Vector{T}} # state prior regularizing cost
    M_observation::Diagonal{T, Vector{T}} # observation measurement regularizing cost
    P_parameters::T # parameters prior regularizing cost
    M_point_cloud::T # point cloud measurement regularizing cost
end

function unpack_dynamics(θ)
    mass = θ[1]
    inertia = θ[2]
    friction_coefficient = θ[3]
    return mass, inertia, friction_coefficient
end

## optimizer
struct CvxOptimizer1310{T} <: Optimizer{T}
    context::CvxContext1310{T}
    objective::CvxObjective1310{T}
end

################################################################################
# methods
################################################################################
function simulate!(context::Context, state::Vector, H::Int)
    mechanism = context.mechanism
    nu = mechanism.dimensions.input
    u = zeros(nu)

    measurements = Vector{Measurement}()
    for i = 1:H
        dynamics(state, mechanism, state, u, nothing)
        measurement = measure(context, state)
        push!(measurements, measurement)
    end
    return measurements
end

function measure(context::CvxContext1310, state::Vector)
    mechanism = context.mechanism
    z = deepcopy(state)
    d = get_point_cloud(mechanism, context.cameras, state)
    measurement = CvxMeasurement1310(z, d)
    return measurement
end

function get_point_cloud(mechanism::Mechanism1170, cameras::Vector{Camera1310{T}}, z) where T
    d = Vector{Matrix{T}}()
    for camera in cameras
        nr = length(camera.camera_rays)
        di = zeros(T, 2, nr)
        A, b, o = get_halfspaces(mechanism, z)
        A, b, o = add_floor(A, b, o)
        julia_point_cloud!(di, camera.eye_position, camera.camera_rays, camera.softness, A, b, o)
        push!(d, di) # TODO maybe this is not copyng the data correctly
    end
    return d
end

function get_halfspaces(mechanism::Mechanism1170, z::Vector{T}) where T
    A = Vector{Matrix{T}}()
    b = Vector{Vector{T}}()
    o = Vector{Vector{T}}()

    set_current_state!(mechanism, z)
    for body in mechanism.bodies
        for shape in body.shapes
            Ai, bi, oi = halfspace_transformation(body.pose, shape.A, shape.b, shape.o)
            push!(A, Ai)
            push!(b, bi)
            push!(o, oi)
        end
    end
    return A, b, o
end

function set_halfspaces!(mechanism::Mechanism1170,
        A::Vector{<:Matrix}, b::Vector{<:Vector}, o::Vector{<:Vector}) where T

    bodies = mechanism.bodies
    contacts = mechanism.contacts

    i = 0
    for body in bodies
        for shape in body.shapes
            i += 1
            shape.A .= A[i]
            shape.b .= b[i]
            shape.o .= o[i]
            contacts[i].A_parent_collider .= A[i]
            contacts[i].b_parent_collider .= b[i] + A[i] * o[i]
        end
    end
    return nothing
end

# A = [ones(4,2)]
# b = [ones(4)]
# o = [ones(2)]
# set_halfspaces!(mech, A, b, o)
# z = zeros(mech.dimensions.state)
# A1, b1, o1 = get_halfspaces(mech, z)
# norm.(A1 - A)[1]
# norm.(b1 - b)[1]
# norm.(o1 - o)[1]

function process_model(context::CvxContext1310, state::Vector, params::Vector)
    set_parameters!(context, params)

    mechanism = context.mechanism
    nz = mechanism.dimensions.state
    nu = mechanism.dimensions.input
    nθ = mechanism.dimensions.parameters
    num_parameters = length(params)

    input = zeros(nu)
    next_state = zeros(nz)
    # jacobian_state = zeros(nz, nz)
    jacobian_parameters = zeros(nz, nθ)

    dynamics(next_state, mechanism, state, input, nothing)
    # dynamics_jacobian_state(jacobian_state, mechanism, state, input, nothing)
    dynamics_jacobian_parameters(jacobian_parameters, mechanism, state, input, nothing)

    function parameters_mapping(context::CvxContext1310, params::Vector)
        set_parameters!(context, params)
        return context.mechanism.parameters
    end
    parameters_jacobian = FiniteDiff.finite_difference_jacobian(params -> parameters_mapping(context, params), params)

    # return next_state, jacobian_state, jacobian_parameters * parameters_jacobian
    return next_state, jacobian_parameters * parameters_jacobian
end


# mech.dimensions
# mech.solver.dimensions
# nz = mech.dimensions.state
# nu = mech.dimensions.input
# nθ = mech.dimensions.parameters
# dθ = zeros(nz, nθ)
# z = rand(nz)
# u = rand(nu)
# θ = rand(nθ)
# dynamics_jacobian_parameters(dθ, mech, z, u, nothing)
# plot(Gray.(abs.(Matrix(dθ))))

function measurement_model(context::CvxContext1310, state::Vector, params::Vector{T}) where T
    pose = state[1:3] # TODO this is not safe, we need to make sure these dimensions are the pose dimensions

    mass, inertia, friction_coefficient, Ab, bb, ob = unpack(params, context)
    Aw, bw, ow = halfspace_transformation(pose, Ab, bb, ob)
    Aw, bw, ow = add_floor(Aw, bw, ow)

    d = Vector{Matrix{T}}()
    for camera in context.cameras
        di = julia_point_cloud(camera.eye_position, camera.camera_rays, camera.softness, Aw, bw, ow)
        push!(d, di)
    end
    measurement = CvxMeasurement1310(state, d)
    return measurement
end

function julia_point_cloud(context::CvxContext1310, state::Vector, params::Vector,
        camera_idx::Int, ray_idx::Int) where T
    pose = state[1:3] # TODO this is not safe, we need to make sure these dimensions are the pose dimensions

    mass, inertia, friction_coefficient, Ab, bb, ob = unpack(params, context)
    Aw, bw, ow = halfspace_transformation(pose, Ab, bb, ob)
    Aw, bw, ow = add_floor(Aw, bw, ow)

    camera = context.cameras[camera_idx]
    d = julia_intersection(camera.eye_position, camera.camera_rays[ray_idx], camera.softness, Aw, bw, ow)
    return d
end

function julia_point_cloud_jacobian_state(context::CvxContext1310, state::Vector, params::Vector{T},
        camera_idx::Int, ray_idx::Int) where T
    ForwardDiff.jacobian(state -> julia_point_cloud(context, state, params, camera_idx, ray_idx), state)
end

function julia_point_cloud_jacobian_parameters(context::CvxContext1310, state::Vector, params::Vector{T},
        camera_idx::Int, ray_idx::Int) where T
    ForwardDiff.jacobian(params -> julia_point_cloud(context, state, params, camera_idx, ray_idx), params)
end

function get_parameters(context::CvxContext1310)
    mechanism = context.mechanism

    z = zeros(mechanism.dimensions.state)
    A, b, o = get_halfspaces(mechanism, z)
    θ_geometry, polytope_dimensions = pack_halfspaces(A, b, o)

    mass = mechanism.bodies[1].mass[1]
    inertia = mechanism.bodies[1].inertia[1]
    friction_coefficient = mechanism.contacts[1].friction_coefficient[1]

    θ_dynamics = [mass; inertia; friction_coefficient]
    params = [θ_dynamics; θ_geometry]
    return params
end

function set_parameters!(context::CvxContext1310, params::Vector)
    mechanism = context.mechanism
    bodies = mechanism.bodies
    contacts = mechanism.contacts

    # set geometry
    θ_geometry = params[4:end]
    A, b, o = unpack_halfspaces(θ_geometry, context.polytope_dimensions)
    set_halfspaces!(mechanism, A, b, o)

    # set dynamics
    θ_dynamics = params[1:3]
    mass, inertia, friction_coefficient = unpack_dynamics(θ_dynamics)
    # TODO we need to generalize to many contacts and bodies
    bodies[1].mass[1] = mass
    bodies[1].inertia[1] = inertia
    contacts[1].friction_coefficient[1] = friction_coefficient

    solver_parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)
    set_parameters!(mechanism.solver, solver_parameters)
    return nothing
end

# # initial guess
# params0 = get_parameters(context)
# params0.θ[1:end] .= rand(17)
# set_parameters!(context, params0)
# params1 = get_parameters(context)
# norm(params0.θ - params1.θ)


################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.1

mech = get_polytope_drop(;
# mech = get_bundle_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    options=Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=false,
        complementarity_correction=0.5,
        )
    );

################################################################################
# test simulation
################################################################################
xp2 = [+0.0,1.5,-0.25]
vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]
H0 = 20
storage = simulate!(mech, z0, H0)
zf = get_current_state(mech)

eye_position = [-0.0, 3.0]
num_points = 50
camera_rays = range(-0.3π, -0.7π, length=num_points)
look_at = [0.0, 0.0]
softness = 100.0
cameras = [Camera1310(eye_position, camera_rays, look_at, softness)]
context = CvxContext1310(mech, cameras)
state = z0

measurements = simulate!(context, state, H0)
vis, anim = visualize!(vis, context, measurements)

################################################################################
# test filtering
################################################################################
function filtering_objective(context::CvxContext1310, obj::CvxObjective1310,
        state::Vector, params::Vector, state_prior::Vector, params_prior::Vector,
        measurement::CvxMeasurement1310)

    z1 = state
    θ1 = params
    θ0 = params_prior
    ẑ1 = measurement.z
    d̂1 = measurement.d
    nθ = length(θ0)

    # initialization
    c = 0.0
    g = zeros(nθ)
    # H = zeros(nθ, nθ)

    # process model
    z̄1, dz̄1dθ1 = process_model(context, state_prior, params)
    # measurement model
    predicted_measurement = measurement_model(context, state, params)
    d1 = predicted_measurement.d

    c = objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1)

    dcdz1 = ForwardDiff.gradient(z1 -> objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1), z1)
    dcdθ1 = ForwardDiff.gradient(θ1 -> objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1), θ1)
    dcdz̄1 = ForwardDiff.gradient(z̄1 -> objective_function(obj, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1), z̄1)

    DcDz1 = dcdz1
    DcDθ1 = dcdθ1 + dz̄1dθ1' * dcdz̄1

    num_cameras = length(d1)
    for i = 1:num_cameras
        camera = context.cameras[i]
        num_rays = size(d1[i], 2)
        for j = 1:num_rays
            di = d1[i][:,j]
            d̂i = d̂1[i][:,j]
            dcdd1 = ForwardDiff.gradient(di -> point_cloud_objective_function(obj, di, d̂i), di)
            dd1dz1 = julia_point_cloud_jacobian_state(context, state, params, i, j)
            dd1dθ1 = julia_point_cloud_jacobian_parameters(context, state, params, i, j)
            DcDz1 += dd1dz1' * dcdd1
            DcDθ1 += dd1dθ1' * dcdd1
        end
    end
    g = [DcDz1; DcDθ1]

    # DcD2z1 = dcd2z1 + dd1dz1' * dcd2d1 * dd1dz1
    # DcD2θ1 = dcd2θ1 +
    # DcDz1θ1 = dcdz1θ1 +
    # H = [DcD2z1 DcDz1θ1; DcDz1θ1' DcD2θ1]
    return c, g#, H
end

function objective_function(obj::CvxObjective1310, z1, θ1, θ0, z̄1, ẑ1, d1, d̂1)
    c = 0.0
    # prior parameters cost
    c += 0.5 * obj.P_parameters * (θ1 - θ0)' * (θ1 - θ0)
    # prior state cost (process model = dynamics)
    c += 0.5 * (z1 - z̄1)' * obj.P_state * (z1 - z̄1)
    # measurement state cost (measurement model = identity)
    c += 0.5 * (z1 - ẑ1)' * obj.M_observation * (z1 - ẑ1)
    # measurement parameter cost (measurement model = point cloud)
    num_cameras = length(d1)
    for i = 1:num_cameras
        num_rays = size(d1[i],2)
        for j = 1:num_rays
            di = d1[i][:,j]
            d̂i = d̂1[i][:,j]
            c += point_cloud_objective_function(obj, di, d̂i)
        end
    end
    return c
end

function point_cloud_objective_function(obj::CvxObjective1310, d::Vector, d̂::Vector)
    0.5 * obj.M_point_cloud * (d - d̂)' * (d - d̂)
end


P_state = Diagonal(ones(6))
M_observation = Diagonal(ones(6))
P_parameters = 1.0
M_point_cloud = 1.0
obj = CvxObjective1310(P_state, M_observation, P_parameters, M_point_cloud)


state_prior = deepcopy(zf)
state_guess = deepcopy(zf)
params_prior = get_parameters(context)
params_guess = get_parameters(context)

c, g = filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, measurements[end])

# @benchmark filtering_objective(context, obj, state_guess, params_guess,
    # state_prior, params_prior, measurements[end])

# Main.@profiler [filtering_objective(context, obj, state_guess, params_guess,
    # state_prior, params_prior, measurements[end]) for i=1:1000]




################################################################################
# filtering demo
################################################################################

# context
context = deepcopy(CvxContext1310(mech, cameras))

# objective
P_state = Diagonal(1e-1ones(6))
M_observation = 1e-1Diagonal(ones(6))
P_parameters = 1e-1
M_point_cloud = 1.0
obj = CvxObjective1310(P_state, M_observation, P_parameters, M_point_cloud)

# state_prior
state_prior = CvxState1310(zf + 3e-1 * (rand(6) .- 0.5))
# params_prior
params_prior = get_parameters(context)
params_prior.θ[4:end] .+= 3e-1 * (rand(14) .- 0.5)

# state_guess
state_guess= CvxState1310(zf)
# params_guess
params_guess = get_parameters(context)
# params_guess.θ[4:end] .+= 0.01

# state_truth
state_truth = CvxState1310(zf)
# params_truth
params_truth= get_parameters(context)

# measurement
measurement = measurement_model(context, state_truth, params_truth)

filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, measurement)

function local_filtering_objective(x)
    nz = length(state_prior.z)
    nθ = length(params_prior.θ)
    state_guess = CvxState1310(x[1:nz])
    params_guess = CvxParameters1310(x[nz .+ (1:nθ)],
        params_prior.num_polytopes,
        params_prior.polytope_dimensions)

    c = filtering_objective(context, obj, state_guess, params_guess,
        state_prior, params_prior, measurement)
    return c
end

function local_filtering_gradient(x)
    FiniteDiff.finite_difference_gradient(x -> local_filtering_objective(x), x)
end

function local_filtering_hessian(x)
    FiniteDiff.finite_difference_hessian(x -> local_filtering_objective(x), x)
end

x0 = [state_prior.z; params_prior.θ]# + 100e-2*(rand(23) .- 0.5)
local_filtering_objective(x0)
local_filtering_gradient(x0)
local_filtering_hessian(x0)

local_projection = x -> x
local_clamping = x -> x
local_D = Diagonal(1e2ones(23))

xsol, xtrace = newton_solver!(x0,
    local_filtering_objective,
    local_filtering_gradient,
    local_filtering_hessian,
    local_projection,
    local_clamping,
    D=local_D,
    residual_tolerance=1e-4,
    reg_min=1e-6,
    reg_max=1e+0,
    reg_step=2.0,
    max_iterations=30,
    )

plot(hcat(xtrace...)')


# vis = Visualizer()
# open(vis)
# set_floor!(vis)
# set_light!(vis)
# set_background!(vis)

anim = MeshCat.Animation(10)
for (i,x) in enumerate(xtrace)
    mechanism = context.mechanism
    nz = context.mechanism.dimensions.state
    zi = x[1:nz]
    θi = x[nz+1:end]
    state_i = CvxState1310(zi)
    params_i = CvxParameters1310(θi, 1, [4])
    measurement_i = measurement_model(context, state_i, params_i)

    set_parameters!(context, params_i)
    build_mechanism!(vis[Symbol(i)], mechanism, name=:mechanism, show_contact=false)
    set_mechanism!(vis[Symbol(i)], mechanism, zi, name=:mechanism)
    build_point_cloud!(vis[Symbol(i)], [num_points], name=:point_cloud)
    set_2d_point_cloud!(vis[Symbol(i)], [eye_position], measurement_i.p, name=:point_cloud)
end
for i = 1:length(xtrace)
    atframe(anim, i) do
        for ii = 1:length(xtrace)
            setvisible!(vis[Symbol(ii)], ii == i)
        end
    end
end
build_point_cloud!(vis[:truth], [num_points], name=:point_cloud, color=RGBA(1,0,0,1.0))
set_2d_point_cloud!(vis[:truth], [eye_position], measurement.p, name=:point_cloud)
set_parameters!(context, params_truth)
build_mechanism!(vis[:truth], context.mechanism, name=:mechanism, color=RGBA(1,0,0,1.0), show_contact=false)
set_mechanism!(vis[:truth], context.mechanism, state_truth.z, name=:mechanism)
settransform!(vis[:truth], MeshCat.Translation(-0.2,0,0))

MeshCat.setanimation!(vis, anim)

# function set_state!(context::CvxContext1310, state::CvxState1310)
#     set_current_state!(context.mechanism, state.z)
#     return nothing
# end

# RobotVisualizer.convert_frames_to_video_and_gif("dyn_informed_training_second")
