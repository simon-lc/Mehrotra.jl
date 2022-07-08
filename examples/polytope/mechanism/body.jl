################################################################################
# body
################################################################################
struct Body171{T,D}
    name::Symbol
    node_index::NodeIndices171
    pose::Vector{T}
    velocity::Vector{T}
    input::Vector{T}
    gravity::Vector{T}
    timestep::Vector{T}
    mass::Vector{T}
    inertia::Matrix{T}
    A_colliders::Vector{Matrix{T}} #polytope
    b_colliders::Vector{Vector{T}} #polytope
end

function Body171(timestep, mass, inertia::Matrix,
        A_colliders::Vector{Matrix{T}},
        b_colliders::Vector{Vector{T}};
        gravity=-9.81,
        name::Symbol=:body,
        node_index::NodeIndices171=NodeIndices171()) where T

    D = size(A_colliders[1],2)
    @assert D == 2

    return Body171{T,D}(
        name,
        node_index,
        zeros(D+1),
        zeros(D+1),
        zeros(D+1),
        [gravity],
        [timestep],
        [mass],
        inertia,
        A_colliders,
        b_colliders,
    )
end

function variable_dimension(body::Body171{T,D}) where {T,D}
    if D == 2
        nv = 3 # velocity
        nx = nv
    else
        error("no 3D yet")
    end
    return nx
end

equality_dimension(body) = variable_dimension(body)

function parameter_dimension(body::Body171{T,D}) where {T,D}
    if D == 2
        nq = 3 # configuration
        nv = 3 # velocity
        nu = 3 # input
        n_gravity = 1 # mass
        n_timestep = 1 # mass
        n_mass = 1 # mass
        n_inertia = 1 # inertia
        nθ = nq + nv + nu + n_gravity + n_timestep + n_mass + n_inertia
    else
        error("no 3D yet")
    end
    return nθ
end

function unpack_body_variables(x::Vector{T}) where T
    v = x
    return v
end

function get_parameters(body::Body171{T,D}) where {T,D}
    @assert D == 2
    pose = body.pose
    velocity = body.velocity
    input = body.input

    gravity = body.gravity
    timestep = body.timestep
    mass = body.mass
    inertia = body.inertia
    θ = [pose; velocity; input; gravity; timestep; mass; inertia[1]]
    return θ
end

function set_parameters!(body::Body171{T,D}, θ) where {T,D}
    @assert D == 2
    off = 0
    body.pose .= θ[off .+ (1:D+1)]; off += D+1
    body.velocity .= θ[off .+ (1:D+1)]; off += D+1
    body.input .= θ[off .+ (1:D+1)]; off += D+1

    body.gravity .= θ[off .+ (1:1)]; off += 1
    body.timestep .= θ[off .+ (1:1)]; off += 1
    body.mass .= θ[off .+ (1:1)]; off += 1
    body.inertia[1,1] = θ[off .+ 1]; off += 1
    return nothing
end

function unpack_body_parameters(θ::Vector, body::Body171{T,D}) where {T,D}
    @assert D == 2
    off = 0
    pose = θ[off .+ (1:D+1)]; off += D+1
    velocity = θ[off .+ (1:D+1)]; off += D+1
    input = θ[off .+ (1:D+1)]; off += D+1

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ 1] * ones(1,1); off += 1
    return pose, velocity, input, timestep, gravity, mass, inertia
end

function body_residual!(e, x, θ, body::Body171)
    node_index = body.node_index
    # variables = primals = velocity
    v25 = unpack_body_variables(x[node_index.x])
    # parameters
    p2, v15, u, timestep, gravity, mass, inertia = unpack_body_parameters(θ[node_index.θ], body)
    # integrator
    p1 = p2 - timestep[1] * v15
    p3 = p2 + timestep[1] * v25
    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])
    # dynamics
    dynamics = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    e[node_index.e] .+= dynamics
    return nothing
end
