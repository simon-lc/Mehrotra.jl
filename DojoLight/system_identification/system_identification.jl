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
    p::Vector{Matrix{T}} # point cloud obtained from several cameras
end

## state
struct CvxState1310{T} <: State{T}
    z::Vector{T}
end

## parameters
struct CvxParameters1310{T} <: Parameters{T}
    θ::Vector{T}
    num_polytopes::Int
    polytope_dimensions::Vector{Int}
end

function unpack(params::CvxParameters1310)
    θ = params.θ
    polytope_dimensions = params.polytope_dimensions

    off = 0
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ (1:1)]; off += 1
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    # center_of_mass = θ[off .+ (1:2)]; off += 2
    A, b, o = unpack_halfspaces(θ[off + 1:end], polytope_dimensions)

    # return mass, inertia, friction_coefficient, center_of_mass, A, b, o
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
    # center_of_mass = θ[4:5]
    # return mass, inertia, friction_coefficient, center_of_mass
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

function measure(context::CvxContext1310, state::CvxState1310)
    mechanism = context.mechanism
    m = deepcopy(state.z)
    p = get_point_cloud(mechanism, context.cameras, state.z)
    measurement = CvxMeasurement1310(m, p)
    return measurement
end

function get_point_cloud(mechanism::Mechanism1170, cameras::Vector{Camera1310{T}}, z) where T
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
        julia_point_cloud!(pi, camera.eye_position, camera.camera_rays, camera.softness, A, b, o)
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

function set_halfspaces!(mechanism::Mechanism1170,
        A::Vector{<:Matrix}, b::Vector{<:Vector}, o::Vector{<:Vector}) where T

    bodies = mechanism.bodies
    contacts = mechanism.contacts
    body = bodies[1]

    for (i, shape) in enumerate(body.shapes)
        shape.A .= A[i]
        shape.b .= b[i]
        shape.o .= o[i]
        contacts[i].A_parent_collider .= A[i]
        contacts[i].b_parent_collider .= b[i] + A[i] * o[i]
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

function process_model(context::CvxContext1310, state::CvxState1310, params::CvxParameters1310)
    set_parameters!(context, params)

    mechanism = context.mechanism
    z0 = state.z
    u = zeros(mechanism.dimensions.input)
    z1 = zeros(mechanism.dimensions.state)
    dynamics(z1, mechanism, z0, u, nothing)

    next_state = CvxState1310(z1)
    return next_state
end

function measurement_model(context::CvxContext1310, state::CvxState1310, params::CvxParameters1310{T}) where T
    z = state.z
    pose = z[1:3] # TODO this is not safe, we need to make sure these dimensions are the pose dimensions

    # mass, inertia, friction_coefficient, center_of_mass, Ab, bb, ob = unpack(params)
    mass, inertia, friction_coefficient, Ab, bb, ob = unpack(params)
    Aw, bw, ow = halfspace_transformation(pose, Ab, bb, ob)
    Aw, bw, ow = add_floor(Aw, bw, ow)

    p = Vector{Matrix{T}}()
    for camera in context.cameras
        pi = julia_point_cloud(camera.eye_position, camera.camera_rays, camera.softness, Aw, bw, ow)
        push!(p, pi)
    end
    measurement = CvxMeasurement1310(state.z, p)
    return measurement
end


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
state = CvxState1310(z0)

measurements = simulate!(context, state, H0)
vis, anim = visualize!(vis, context, measurements)

################################################################################
# test filtering
################################################################################
function get_parameters(context::CvxContext1310)
    mechanism = context.mechanism

    z = zeros(mechanism.dimensions.state)
    A, b, o = get_halfspaces(mechanism, z)
    θ_geometry, polytope_dimensions = pack_halfspaces(A, b, o)
    num_polytopes = length(polytope_dimensions)

    mass = mechanism.bodies[1].mass[1]
    inertia = mechanism.bodies[1].inertia[1]
    friction_coefficient = mechanism.contacts[1].friction_coefficient[1]
    # center_of_mass = [0.0, 0.0]

    # θ_dynamics = [mass; inertia; friction_coefficient; center_of_mass]
    θ_dynamics = [mass; inertia; friction_coefficient]
    θ = [θ_dynamics; θ_geometry]
    return CvxParameters1310(θ, num_polytopes, polytope_dimensions)
end

function set_parameters!(context::CvxContext1310, params::CvxParameters1310)
    polytope_dimensions = params.polytope_dimensions
    np = length(polytope_dimensions)
    mechanism = context.mechanism
    bodies = mechanism.bodies
    contacts = mechanism.contacts
    θ = params.θ

    # set geometry
    θ_geometry = θ[4:end]
    A, b, o = unpack_halfspaces(θ_geometry, polytope_dimensions)
    set_halfspaces!(mechanism, A, b, o)

    # set dynamics
    θ_dynamics = θ[1:3]
    # mass, inertia, friction_coefficient, center_of_mass = unpack_dynamics(θ_dynamics)
    mass, inertia, friction_coefficient = unpack_dynamics(θ_dynamics)
    # TODO we need to generalize to many contacts and bodies
    bodies[1].mass[1] = mass
    bodies[1].inertia[1] = inertia
    contacts[1].friction_coefficient[1] = friction_coefficient
    # TODO center of mass is ignored at the moment

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


function filtering_objective(context::CvxContext1310, obj::CvxObjective1310,
        state::CvxState1310, params::CvxParameters1310,
        state_prior::CvxState1310, params_prior::CvxParameters1310,
        measurement::CvxMeasurement1310)

    θ0 = params_prior.θ
    z1 = state.z
    θ1 = params.θ
    z̄1 = measurement.z
    p̄1 = measurement.p
    nθ = length(θ0)

    # initialization
    c = 0.0
    g = zeros(nθ)
    # H = zeros(nθ, nθ)

    # process model
    predicted_state = process_model(context, state_prior, params)
    # measurement model
    predicted_measurement = measurement_model(context, state, params)

    c = objective_function(obj, z1, predicted_state.z, z̄1, θ0, θ1, predicted_measurement.p, p̄1)
    DcDz1 = dcdz1 + dz1_preddz1' * dcdz1_pred
    DcDθ1 = dcdθ1 + dz1_preddθ1' * dcdz1_pred + dp1_preddθ1' * dcdp1_pred
    g = [DcDz1; DcDθ1]
    DcD2z1 = dcd2z1
    DcD2θ1 = dcd2θ1 + dθ1(dp1_preddθ1' * dcdp1_pred)
    DcDz1θ1 = dcdz1θ1 + dθ1(dz1_preddz1' * dcdz1_pred)
    return c, g#, H
end

function objective_function(obj::CvxObjective1310, z1, z1_pred, z̄1, θ0, θ1, p1_pred, p̄1)
    c = 0.0
    # prior state cost (process model = dynamics)
    c += 0.5 * (z1 - z1_pred)' * obj.P_state * (z1 - z1_pred)
    # measurement state cost (measurement model = identity)
    c += 0.5 * (z1 - z̄1)' * obj.M_observation * (z1 - z̄1)
    # prior parameters cost
    c += 0.5 * obj.P_parameters * (θ0 - θ1)' * (θ0 - θ1)
    # measurement parameter cost (measurement model = point cloud)
    for (i, p̄i) in enumerate(p̄1)
        pi = p1_pred[i]
        c += 0.5 * obj.M_point_cloud * norm(pi - p̄i)^2
    end
    return c
end


P_state = Diagonal(ones(6))
M_observation = Diagonal(ones(6))
P_parameters = 1.0
M_point_cloud = 1.0
obj = CvxObjective1310(P_state, M_observation, P_parameters, M_point_cloud)

state_prior = CvxState1310(zf)
state_guess = CvxState1310(zf)
params_prior = get_parameters(context)
params_guess = get_parameters(context)

c = filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, measurements[end])

# @benchmark filtering_objective(context, obj, state_guess, params_guess,
#     state_prior, params_prior, measurements[end])
#
# Main.@profiler [filtering_objective(context, obj, state_guess, params_guess,
#     state_prior, params_prior, measurements[end]) for i=1:1000]




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

function mysolve!(θinit, loss, Gloss, Hloss, projection, clamping;
        max_iterations=20,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=2.0,
        residual_tolerance=1e-4,
        D=Diagonal(ones(length(θinit))))

    θ = deepcopy(θinit)
    trace = [deepcopy(θ)]
    reg = reg_max

    # newton's method
    for iterations = 1:max_iterations
        l = loss(θ)
        (l < residual_tolerance) && break
        G = Gloss(θ)
        H = Hloss(θ)

        # reg = clamp(norm(G, Inf)/10, reg_min, reg_max)
        Δθ = - (H + reg * D) \ G

        # linesearch
        α = 1.0
        for j = 1:10
            l_candidate = loss(projection(θ + clamping(α * Δθ)))
            if l_candidate <= l
                reg = clamp(reg/reg_step, reg_min, reg_max)
                break
            end
            α /= 2
            if j == 10
                reg = clamp(reg*reg_step, reg_min, reg_max)
            end
        end

        # header
        if rem(iterations - 1, 10) == 0
            @printf "-------------------------------------------------------------------\n"
            @printf "iter   loss        step        |step|∞     |grad|∞     reg         \n"
            @printf "-------------------------------------------------------------------\n"
        end
        # iteration information
        @printf("%3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e\n",
            iterations,
            l,
            mean(α),
            norm(clamping(α * Δθ), Inf),
            norm(G, Inf),
            reg,
            )
        θ = projection(θ + clamping(α * Δθ))
        push!(trace, deepcopy(θ))
    end
    return θ, trace
end

xsol, xtrace = mysolve!(x0,
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
