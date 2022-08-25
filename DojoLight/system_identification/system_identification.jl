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
include("../system_identification/methods.jl")
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
# demo
################################################################################
timestep = 0.20;
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
softness = 2.5
cameras = [Camera1310(eye_position, camera_rays, look_at, softness)]
context = CvxContext1310(mech, cameras)
state = z0

measurements = simulate!(context, state, H0)
# vis, anim = visualize!(vis, context, measurements)

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
    nz = length(z1)
    nθ = length(θ0)

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

    # DcD2z1 = dcd2z1# + dd1dz1' * dcd2d1 * dd1dz1
    # DcD2θ1 = dcd2θ1# +
    # DcDz1θ1 = dcdz1θ1# +
    # H = [DcD2z1 DcDz1θ1; DcDz1θ1' DcD2θ1]
    H = Diagonal([diag(obj.P_state + obj.M_observation); obj.P_parameters * ones(nθ)])
    return c, g, H
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

c, g, H = filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, measurements[end])

# @benchmark filtering_objective(context, obj, state_guess, params_guess,
#     state_prior, params_prior, measurements[end])

# Main.@profiler [filtering_objective(context, obj, state_guess, params_guess,
#     state_prior, params_prior, measurements[end]) for i=1:100]




################################################################################
# filtering demo
################################################################################

# context
context = deepcopy(CvxContext1310(mech, cameras))

# objective
P_state = Diagonal(1e+2ones(6))
M_observation = 1e-2Diagonal(ones(6))
P_parameters = 3e-3
M_point_cloud = 1e-0
obj = CvxObjective1310(P_state, M_observation, P_parameters, M_point_cloud)

# prior
state_prior = deepcopy(zf + 2e-1 * (rand(6) .- 0.5))
params_prior = get_parameters(context)
params_prior[4:end] .+= 5e-1 * (rand(14) .- 0.5)

# guess
state_guess = deepcopy(zf)
params_guess = get_parameters(context)
# params_guess.θ[4:end] .+= 0.01

# truth
state_truth = deepcopy(zf)
params_truth = get_parameters(context)

# measurement
measurement = measurement_model(context, state_truth, params_truth)
measurement.z .+=  5e-2(rand(6) .- 0.5)


filtering_objective(context, obj, state_guess, params_guess,
    state_prior, params_prior, measurement)

function local_filtering_objective(x)
    nz = length(state_prior)
    nθ = length(params_prior)
    state_guess = deepcopy(x[1:nz])
    params_guess = deepcopy(x[nz .+ (1:nθ)])

    c, g, H = filtering_objective(context, obj, state_guess, params_guess,
        state_prior, params_prior, measurement)
    return c
end

function local_filtering_gradient(x)
    nz = length(state_prior)
    nθ = length(params_prior)
    state_guess = deepcopy(x[1:nz])
    params_guess = deepcopy(x[nz .+ (1:nθ)])

    c, g, H = filtering_objective(context, obj, state_guess, params_guess,
        state_prior, params_prior, measurement)
    return g
end

function local_filtering_hessian(x)
    nz = length(state_prior)
    nθ = length(params_prior)
    state_guess = deepcopy(x[1:nz])
    params_guess = deepcopy(x[nz .+ (1:nθ)])

    c, g, H = filtering_objective(context, obj, state_guess, params_guess,
        state_prior, params_prior, measurement)
    return H
end

x0 = [state_prior; params_prior]# + 100e-2*(rand(23) .- 0.5)
local_filtering_objective(x0)
local_filtering_gradient(x0)
local_filtering_hessian(x0)

function projection(context::CvxContext1310, x;
        bound_mass=[1e-1, 1e1],
        bound_inertia=[1e-1, 1e1],
        bound_friction_coefficient=[0.0, 2.0],
        bound_b=[5e-2, 1e0],
        bound_o=[-3.0, 3.0],
        )
    nz = context.mechanism.dimensions.state
    state = deepcopy(x[1:nz])
    params = deepcopy(x[nz+1:end])
    mass, inertia, friction_coefficient, A, b, o = unpack(params, context)
    mass = clamp(mass, bound_mass...)
    inertia = clamp(inertia, bound_inertia...)
    friction_coefficient = clamp(friction_coefficient, bound_friction_coefficient...)

    for i = 1:length(A)
        for j = 1:size(A[i],1)
            A[i][j,:] ./= norm(A[i][j,:]) + 1e-5
        end
    end
    b = [clamp.(bi, bound_b...) for bi in b]
    o = [clamp.(oi, bound_o...) for oi in o]

    π_params = pack(mass, inertia, friction_coefficient, A, b, o)
    π_x = [state; π_params]
    return π_x
end

local_projection100(x) = projection(context, x)
local_clamping = x -> x

x0
projection(context, x0)
xsol, xtrace = newton_solver!(x0,
    local_filtering_objective,
    local_filtering_gradient,
    local_filtering_hessian,
    local_projection100,
    local_clamping,
    residual_tolerance=1e-4,
    reg_min=1e-0,
    reg_max=1e+3,
    reg_step=2.0,
    max_iterations=30,
    line_search_iterations=10,
    )

# plot(hcat(xtrace...)')


# vis = Visualizer()
# render(vis)
# set_floor!(vis, color=RGBA(0,0,0,0.4))
# set_light!(vis)
# set_background!(vis)

prior = [state_prior; params_prior]
solution = [state_truth; params_truth]
vis, anim = visualize_solve!(vis, context, prior, solution, xtrace)


# open(vis)
