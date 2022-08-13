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
set_floor!(vis)
set_light!(vis)
set_background!(vis)

include("../src/DojoLight.jl")

include("../cvxnet/softmax.jl")
include("../cvxnet/loss.jl")
include("../cvxnet/point_cloud.jl")


################################################################################
# structure
################################################################################
# measurement obtained at each time step
abstract type Measurement{T} end

# time-varying state of the system
abstract type State{T} end

# parameters of the system that are fixed in time and that we want to estimate
abstract type Parameters{T} end

# all ground-truth information including time-invariant parameters that we want to estimate (parameters)
# and that we assume are known (camera position for instance).
# it also contains the state of the system
abstract type Context{T} end

# objective that we want to minimize to regress correct states and parameters
abstract type Objective{T} end

# find the best fitting states and parameters given a sequence of measurement and a prior over the state and parameters
# it contains the context and the objective
abstract type Optimizer{T} end


## measurement
struct CvxMeasurement1290{T} <: Measurement{T}
    # x::Vector{T} # polytope position
    # q::Vector{T} # polytope orientation
    z::Vector{T} # polytope state, we assume the whole state is observed
    p::Vector{Matrix{T}} # point cloud obtained from several cameras
end

## state
struct CvxState1290{T} <: State{T}
    z::Vector{T}
end

function unpack(state::CvxState1290)
    off = 0
    x = z[off .+ (1:3)]; off += 2
    v = z[off .+ (1:2)]; off += 2
    q = z[off .+ (1:1)]; off += 1
    ω = z[off .+ (1:1)]; off += 1
    return x, v, q, ω
end

function pack!(state::CvxState1290, x, v, q, ω)
    state.z .= [x; v; q; ω]
    return nothing
end

## parameters
struct CvxParameters1290{T} <: Parameters{T}
    θ::Vector{T}
    num_polytopes::Int
    polytope_dimensions::Vector{Int}
end

function unpack(params::CvxParameters1290)

    return mass, friction, center_of_mass, A, b, o
end

function pack!(params::CvxParameters1290, mass::T, friction::T, center_of_mass::T,
        A::Vector{<:Matrix}, b::Vector{<:Vector}, o::Vector{<:Vector}) where T

    return nothing
end

# camera
struct Camera1290{T}
    eye_position::Vector{T}
    camera_rays
    look_at::Vector{T}
    softness::T
end

## context
struct CvxContext1290{T} <: Context{T}
    mechanism::Mechanism1170
    cameras::Vector{Camera1290{T}}
end

## objective
struct CvxObjective1290{T} <: Objective{T}
    P_state::Diagonal{T, Vector{T}} # prior regularizing cost
    M_state::Diagonal{T, Vector{T}} # measurement regularizing cost
    P_parameters::T # prior regularizing cost
    M_point_cloud::T # measurement regularizing cost
end

function get_parameters(context::CvxContext1290)
    mechanism = context.mechanism
    z = zeros(mechanism.dimensions.state)
    A, b, o = get_halfspaces(mechanism, z)
    θ, polytope_dimensions = pack_halfspaces(A, b, o)
    num_polytopes = length(polytope_dimensions)
    return CvxParameters1290(θ, num_polytopes, polytope_dimensions)
end


## optimizer
struct CvxOptimizer1290{T} <: Optimizer{T}
    context::CvxContext1290{T}
    objective::CvxObjective1290{T}
end

################################################################################
# methods
################################################################################
function simulate!(context::Context, state::State, H::Int)
    mechanism = context.mechanism
    nu = mechanism.dimensions.input
    u = zeros(nu)

    measurements = Vector{Measurement}()
    for i = 1:H
        dynamics(state.z, mechanism, state.z, u, nothing)
        measurement = measure(context, state)
        push!(measurements, measurement)
    end
    return measurements
end

function measure(context::CvxContext1290, state::CvxState1290)
    mechanism = context.mechanism
    m = deepcopy(state.z)
    p = get_point_cloud(mechanism, context.cameras, state.z)
    measurement = CvxMeasurement1290(m, p)
    return measurement
end

function get_point_cloud(mechanism::Mechanism1170, cameras::Vector{Camera1290{T}}, z) where T
    p = Vector{Matrix{T}}()
    for camera in cameras
        nr = length(camera.camera_rays)
        pi = zeros(T, 2, nr)
        A, b, o = get_halfspaces(mechanism, z)
        # add floor
        Af = [0.0  1.0]
        bf = [0.0]
        of = [0.0, 0.0]
        push!(A, Af)
        push!(b, bf)
        push!(o, of)
        sumeet_point_cloud!(pi, camera.eye_position, camera.camera_rays, camera.softness, A, b, o)
        push!(p, pi)
    end
    return p
end

function halfspace_transformation(pose::Vector{T}, A::Matrix, b::Vector, o::Vector) where T
    position = pose[1:2]
    orientation = pose[3:3]

    wRb = x_2d_rotation(orientation) # rotation from body frame to world frame
    bRw = wRb' # rotation from world frame to body frame
    Ā = A * bRw
    b̄ = b + A * bRw * position
    ō = wRb * o
    return Ā, b̄, ō
end

function halfspace_transformation(pose::Vector{T}, A::Vector{<:Matrix},
        b::Vector{<:Vector}, o::Vector{<:Vector}) where T
    np = length(b)
    Ā = Vector{Matrix{T}}()
    b̄ = Vector{Vector{T}}()
    ō = Vector{Vector{T}}()
    for i = 1:np
        Āi, b̄i, ōi = halfspace_transformation(pose, A[i], b[i], o[i])
        push!(Ā, Āi)
        push!(b̄, b̄i)
        push!(ō, ōi)
    end
    return Ā, b̄, ō
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

include("../system_identification/visuals.jl")


################################################################################
# demo
################################################################################
timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);
friction_coefficient = 0.1

# mech = get_polytope_drop(;
mech = get_bundle_drop(;
    timestep=timestep,
    gravity=gravity,
    mass=mass,
    inertia=inertia,
    friction_coefficient=friction_coefficient,
    method_type=:symbolic,
    # method_type=:finite_difference,
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

eye_position = [-0.0, 3.0]
camera_rays = range(-0.3π, -0.7π, length=50)
look_at = [0.0, 0.0]
softness = 10.0
cameras = [Camera1290(eye_position, camera_rays, look_at, softness)]
context = CvxContext1290(mech, cameras)
state = CvxState1290(z0)

measurements = simulate!(context, state, H0)
vis, anim = visualize!(vis, context, measurements)


function filtering_objective(context::CvxContext1290, obj::CvxObjective1290,
        state::CvxState1290, params::CvxParameters1290,
        state_prior::CvxState1290, params_prior::CvxParameters1290,
        measurement::CvxMeasurement1290)

    mechanism = context.mechanism
    cameras = context.cameras
    z0 = state_prior.z
    θ0 = params_prior.θ
    z1 = state.z
    θ1 = params.θ
    polytope_dimensions = params.polytope_dimensions
    A1, b1, o1 = unpack_halfspaces(θ1, polytope_dimensions)
    z̄1 = measurement.z
    p̄1 = measurement.p

    # process model
    u = zeros(mechanism.dimensions.input)
    z1_pred = zeros(mechanism.dimensions.state)
    dynamics(z1_pred, mechanism, z0, u, nothing)

    # measurement model
    pose1 = z1[1:3] # TODO this is not safe,we need to make sure these dimensions are the pose dimensions
    Ā1, b̄1, ō1 = halfspace_transformation(pose1, A1, b1, o1)


    c = 0.0
    # prior state cost (process model = dynamics)
    c += 0.5 * (z1 - z1_pred)' * obj.P_state * (z1 - z1_pred)
    # measurement state cost (measurement model = identity)
    c += 0.5 * (z1 - z̄1)' * obj.M_state * (z1 - z̄1)

    # prior parameters cost
    c += 0.5 * obj.P_parameters * (θ0 - θ1)' * (θ0 - θ1)
    # measurement parameter cost (measurement model = point cloud)
    for (i, p̄i1) in enumerate(p̄1)
        camera = cameras[i]
        pi1 = sumeet_point_cloud(camera.eye_position, camera.camera_rays, camera.softness, Ā1, b̄1, ō1)
        c += 0.5 * obj.M_point_cloud * norm(pi1 - p̄i1)^2
    end
    return c
end

Pz = Diagonal(ones(6))
Mz = Diagonal(ones(6))
Pp = 1.0
Mp = 1.0
obj = CvxObjective1290(Pz, Mz, Pp, Mp)

state_prior = CvxState1290(z0)
state_guess = CvxState1290(z0)
params_prior = get_parameters(context)
params_guess = get_parameters(context)

c = filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, measurements[1])
