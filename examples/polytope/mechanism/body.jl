################################################################################
# body
################################################################################
struct Body182{T,D} <: Node{T}
    name::Symbol
    index::NodeIndices182
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

function Body182(timestep, mass, inertia::Matrix,
        A_colliders::Vector{Matrix{T}},
        b_colliders::Vector{Vector{T}};
        gravity=-9.81,
        name::Symbol=:body,
        index::NodeIndices182=NodeIndices182()) where T

    D = size(A_colliders[1],2)
    @assert D == 2

    return Body182{T,D}(
        name,
        index,
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

primal_dimension(body::Body182{T,D}) where {T,D} = 3
cone_dimension(body::Body182{T,D}) where {T,D} = 0
variable_dimension(body::Body182{T,D}) where {T,D} = primal_dimension(body) + 2 * cone_dimension(body)
optimality_dimension(body::Body182{T,D}) where {T,D} = primal_dimension(body)
slackness_dimension(body::Body182{T,D}) where {T,D} = cone_dimension(body)
equality_dimension(body::Body182{T,D}) where {T,D} = optimality_dimension(body) + slackness_dimension(body)

function parameter_dimension(body::Body182{T,D}) where {T,D}
    @assert D == 2
    nq = 3 # configuration
    nv = 3 # velocity
    nu = 3 # input
    n_gravity = 1 # mass
    n_timestep = 1 # mass
    n_mass = 1 # mass
    n_inertia = 1 # inertia
    nθ = nq + nv + nu + n_gravity + n_timestep + n_mass + n_inertia
    return nθ
end

function unpack_variables(x::Vector{T}, body::Body182{T}) where T
    return x
end

function get_parameters(body::Body182{T,D}) where {T,D}
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

function set_parameters!(body::Body182{T,D}, θ) where {T,D}
    pose, velocity, input, timestep, gravity, mass, inertia = unpack_parameters(θ, body)
    body.pose .= pose
    body.velocity .= velocity
    body.input .= input

    body.gravity .= gravity
    body.timestep .= timestep
    body.mass .= mass
    body.inertia .= inertia
    return nothing
end

function unpack_parameters(θ::Vector, body::Body182{T,D}) where {T,D}
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

function find_body(bodies::AbstractVector{<:Body182}, name::Symbol)
    idx = findfirst(x -> x == name, getfield.(bodies, :name))
    return bodies[idx]
end

function residual!(e, x, θ, body::Body182)
    index = body.index
    # variables = primals = velocity
    v25 = unpack_variables(x[index.variables], body)
    # parameters
    p2, v15, u, timestep, gravity, mass, inertia = unpack_parameters(θ[index.parameters], body)
    # integrator
    p1 = p2 - timestep[1] * v15
    p3 = p2 + timestep[1] * v25

    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])
    # dynamics
    optimality = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    e[index.optimality] .+= optimality
    return nothing
end

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end
