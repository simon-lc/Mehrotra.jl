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
