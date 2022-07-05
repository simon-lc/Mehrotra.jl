################################################################################
# body
################################################################################
struct Body130{T,D}
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

function Body130(timestep, mass, inertia::Matrix,
        A_colliders::Vector{Matrix{T}}, b_colliders::Vector{Vector{T}};
        gravity=-9.81) where T

    D = size(A_colliders[1])[2]
    @assert D == 2

    return Body130{T,D}(
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

function variable_dimension(body::Body130{T,D}) where {T,D}
    if D == 2
        nv = 3 # velocity
        nx = nv
    else
        error("no 3D yet")
    end
    return nx
end

function unpack_body_variables(x::Vector{T}) where T
    v = x
    return v
end

function parameter_dimension(body::Body130{T,D}) where {T,D}
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

function get_parameters(body::Body130{T,D}) where {T,D}
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

function set_parameters!(body::Body130{T,D}, θ) where {T,D}
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

function unpack_body_parameters(θ::Vector{T}; D::Int=2) where T
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


################################################################################
# contact
################################################################################
struct Contact130{T,D,NP,NC}
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function Contact130(A_parent_collider::Matrix{T}, b_parent_collider::Vector{T},
        A_child_collider::Matrix{T}, b_child_collider::Vector{T}) where {T}
    D = size(A_parent_collider)[2]
    NP = size(A_parent_collider)[1]
    NC = size(A_child_collider)[1]
    return Contact130{T,D,NP,NC}(
        A_parent_collider,
        b_parent_collider,
        A_child_collider,
        b_child_collider,
    )
end

function Contact130(parent_body::Body130, child_body::Body130) where {T}
    return Contact130(
        parent_body.A_colliders[1],
        parent_body.b_colliders[1],
        child_body.A_colliders[1],
        child_body.b_colliders[1],
    )
end

function variable_dimension(contact::Contact130{T,D}) where {T,D}
    if D == 2
        nγ = 2*1 # impact (dual and slack)
        nb = 2*2 # friction (dual and slack)
        nx = nγ + nb
    else
        error("no 3D yet")
    end
    return nx
end

function parameter_dimension(contact::Contact130{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = nAp + nbp + nAc + nbc
    return nθ
end

function get_parameters(contact::Contact130{T,D}) where {T,D}
    θ = [
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::Contact130{T,D,NP,NC}, θ) where {T,D,NP,NC}
    off = 0
    contact.A_parent_collider .= reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    contact.b_parent_collider .= θ[off .+ (1:NP)]; off += NP
    contact.A_child_collider .= reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    contact.b_child_collider .= θ[off .+ (1:NC)]; off += NC
    return nothing
end


################################################################################
# dimensions
################################################################################
struct MechanismDimensions130
    body_configuration::Int
    body_velocity::Int
    body_state::Int
    bodies::Int
    contacts::Int
    variables::Int
    parameters::Int
    primals::Int
    cone::Int
end


################################################################################
# mechanism
################################################################################
struct Mechanism130{T,D,NB}
    variables::Vector{T}
    parameters::Vector{T}
    solver::Solver228{T}
    bodies::Vector{Body130{T}}
    contacts::Vector{Contact130{T}}
    dimensions::MechanismDimensions130
    # contacts::Contact{T}
    # equalities::Vector{Equality{T}}
    # inequalities::Vector{Inequality{T}}
end

function Mechanism130(bodies::Vector, contacts::Vector;
        options::Options228=Options228()) where {T,D}
    # dimensions
    nq = 3 # in 2D
    nv = 3 # in 2D
    nx = 6 # in 2D
    nb = length(bodies)
    nc = length(contacts)
    nx = sum(variable_dimension.(bodies)) + sum(variable_dimension.(contacts))# + num_contacts
    nθ = sum(parameter_dimension.(bodies)) + sum(parameter_dimension.(contacts))# + num_contacts
    num_primals = sum(variable_dimension.(bodies))
    num_cone = Int(sum(variable_dimension.(contacts)) / 2)
    dim = MechanismDimensions130(nq, nv, nx, nb, nc, nx, nθ, num_primals, num_cone)

    # solver
    parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)
    methods = mechanism_methods(bodies, contacts, dim)
    solver = mechanism_solver(parameters, dim, methods, options=options)

    # vectors
    variables = solver.solution.all
    parameters = solver.parameters

    mechanism = Mechanism130{T,D,nb}(
        variables,
        parameters,
        solver,
        bodies,
        dim,
        )
    return mechanism
end

function mechanism_solver(parameter::Vector, dim::MechanismDimensions130,
        methods::ProblemMethods228; options::Options228=Options228())

    num_primals = dim.primals
    num_cone = dim.cone
    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    solver = Solver(
        nothing,
        num_primals,
        num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        methods=methods,
        options=options
    )

    return solver
end


# function step!(mechanism::Mechanism130{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism130{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism130{T})
# end
#
# function set_input!(mechanism::Mechanism130{T})
# end
#
# function set_current_state!(mechanism::Mechanism130{T})
# end
#
# function set_next_state!(mechanism::Mechanism130{T})
# end
#
# function get_current_state!(mechanism::Mechanism130{T})
# end
#
# function get_next_state!(mechanism::Mechanism130{T})
# end


Aa = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
ba = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]

Ab = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bb = 0.5*[
     1,
     1,
     1,
     1,
    ]

timestep = 0.01
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
bodya = Body130(timestep, mass, inertia, [Aa], [ba], gravity=gravity)
bodyb = Body130(timestep, mass, inertia, [Ab], [bb], gravity=gravity)
bodies = [bodya, bodyb]

contacts = [Contact130(bodies[1], bodies[2])]


θbody = get_parameters(bodya)
θcontact = get_parameters(contacts[1])
set_parameters!(bodya, θbody)
set_parameters!(contacts[1], θcontact)



# dimensions
nq = 3 # in 2D
nv = 3 # in 2D
nx = 6 # in 2D
nb = length(bodies)
nc = length(contacts)
nx = sum(variable_dimension.(bodies)) + sum(variable_dimension.(contacts))# + num_contacts
nθ = sum(parameter_dimension.(bodies)) + sum(parameter_dimension.(contacts))# + num_contacts
num_primals = sum(variable_dimension.(bodies))
num_cone = Int(sum(variable_dimension.(contacts)) / 2)
dim = MechanismDimensions130(nq, nv, nx, nb, nc, nx, nθ, num_primals, num_cone)


function mechanism_methods(bodies::Vector, contacts::Vector, dim::MechanismDimensions130)

    function residual!(r, x, θ, node_index)
        # variables = primals = velocity
        v25 = unpack_body_variables(x[node_index.x])
        # parameters
        p2, v15, u, timestep, gravity, mass, inertia = unpack_body_parameters(θ[node_index.θ])
        # integrator
        p1 = p2 - timestep[1] * v15
        p3 = p2 + timestep[1] * v25
        # mass matrix
        M = Diagonal([mass[1]; mass[1]; inertia[1]])
        # dynamics
        dyn = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ua * timestep[1];
        r[node_index.r] .+= dyn
        return nothing
    end

    residuals = [ri!(r, xi, θi) = residual!(r, xi, θi, node_index) for node_index in node_indices]

    # for body in bo

    # function residual_jacobian_variables!(J, x, θ)
    #
    #     return nothing
    # end
    #
    # function residual_jacobian_parameters!(J, x, θ)
    #
    #     return nothing
    # end


    # e = nothing
    # ex = nothing
    # eθ = nothing

    # return methods
end

mechanism_methods(bodies, contacts, dim)

# solver = mechanism_solver(bodies, contacts, dim)
# mech = Mechanism130(bodies, contacts)
