################################################################################
# body
################################################################################
struct QuasistaticObject1160{T,D} <: Body{T}
    name::Symbol
    index::NodeIndices1160
    pose::Vector{T}
    # velocity::Vector{T}
    input::Vector{T}
    gravity::Vector{T}
    timestep::Vector{T}
    mass::Vector{T}
    inertia::Matrix{T}
    # stiffness::Vector{T}
    shapes::Vector
end

function QuasistaticObject1160(timestep::T, mass, inertia::Matrix,
        shapes::Vector;
        gravity=-9.81,
        name::Symbol=:body,
        index::NodeIndices1160=NodeIndices1160()) where T

    D = 2
    return QuasistaticObject1160{T,D}(
        name,
        index,
        zeros(D+1),
        # zeros(D+1),
        zeros(D+1),
        [gravity],
        [timestep],
        [mass],
        inertia,
        shapes,
    )
end

primal_dimension(body::QuasistaticObject1160{T,D}) where {T,D} = 3
cone_dimension(body::QuasistaticObject1160{T,D}) where {T,D} = 0

function parameter_dimension(body::QuasistaticObject1160{T,D}) where {T,D}
    @assert D == 2
    nq = 3 # configuration
    # nv = 3 # velocity
    nu = 3 # input
    n_gravity = 1 # mass
    n_timestep = 1 # mass
    n_mass = 1 # mass
    n_inertia = 1 # inertia
    # nθ = nq + nv + nu + n_gravity + n_timestep + n_mass + n_inertia
    nθ = nq + nu + n_gravity + n_timestep + n_mass + n_inertia
    return nθ
end

function unpack_variables(x::Vector, body::QuasistaticObject1160{T}) where T
    return x
end

function get_parameters(body::QuasistaticObject1160{T,D}) where {T,D}
    @assert D == 2
    pose = body.pose
    # velocity = body.velocity
    input = body.input

    gravity = body.gravity
    timestep = body.timestep
    mass = body.mass
    inertia = body.inertia
    # θ = [pose; velocity; input; gravity; timestep; mass; inertia[1]]
    θ = [pose; input; gravity; timestep; mass; inertia[1]]
    return θ
end

function set_parameters!(body::QuasistaticObject1160{T,D}, θ) where {T,D}
    # pose, velocity, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    pose, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    body.pose .= pose
    # body.velocity .= velocity
    body.input .= input

    body.gravity .= gravity
    body.timestep .= timestep
    body.mass .= mass
    body.inertia .= inertia
    return nothing
end

function unpack_parameters(θ::Vector, body::QuasistaticObject1160{T,D}) where {T,D}
    @assert D == 2
    off = 0
    pose = θ[off .+ (1:D+1)]; off += D+1
    # velocity = θ[off .+ (1:D+1)]; off += D+1
    input = θ[off .+ (1:D+1)]; off += D+1

    gravity = θ[off .+ (1:1)]; off += 1
    timestep = θ[off .+ (1:1)]; off += 1
    mass = θ[off .+ (1:1)]; off += 1
    inertia = θ[off .+ 1] * ones(1,1); off += 1
    # return pose, velocity, input, timestep, gravity, mass, inertia
    return pose, input, timestep, gravity, mass, inertia
end

function unpack_pose_timestep(θ::Vector, body::QuasistaticObject1160{T,D}) where {T,D}
    pose, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    return pose, timestep
end

function find_body(bodies::AbstractVector{<:QuasistaticObject1160}, name::Symbol)
    idx = findfirst(x -> x == name, getfield.(bodies, :name))
    return bodies[idx]
end

function residual!(e, x, θ, body::QuasistaticObject1160)
    index = body.index
    # variables = primals = velocity
    v25 = unpack_variables(x[index.variables], body)
    # parameters
    # p2, v15, u, timestep, gravity, mass, inertia = unpack_parameters(θ[index.parameters], body)
    p2, u, timestep, gravity, mass, inertia = unpack_parameters(θ[index.parameters], body)
    # integrator
    # p1 = p2 - timestep[1] * v15
    p3 = p2 + timestep[1] * v25

    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])

    K = Diagonal(stiffness)
    # dynamics
    optimality = M * v25 - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    e[index.optimality] .+= optimality
    return nothing
end
