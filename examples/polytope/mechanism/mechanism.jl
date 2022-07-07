
# function step!(mechanism::Mechanism170{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism170{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism170{T})
# end
#
# function set_input!(mechanism::Mechanism170{T})
# end
#
# function set_current_state!(mechanism::Mechanism170{T})
# end
#
# function set_next_state!(mechanism::Mechanism170{T})
# end
#
# function get_current_state!(mechanism::Mechanism170{T})
# end
#
# function get_next_state!(mechanism::Mechanism170{T})
# end
using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions


vis = Visualizer()
render(vis)

include("../contact_model/lp_2d.jl")
include("../visuals.jl")
include("../rotate.jl")
include("../quaternion.jl")

mutable struct NodeIndices170
    e::Vector{Int}
    x::Vector{Int}
    θ::Vector{Int}
end

function NodeIndices170()
    return NodeIndices170(
        collect(1:0),
        collect(1:0),
        collect(1:0),
    )
end

################################################################################
# body
################################################################################
struct Body170{T,D}
    name::Symbol
    node_index::NodeIndices170
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

function Body170(timestep, mass, inertia::Matrix,
        A_colliders::Vector{Matrix{T}},
        b_colliders::Vector{Vector{T}};
        gravity=-9.81,
        name::Symbol=:body,
        node_index::NodeIndices170=NodeIndices170()) where T

    D = size(A_colliders[1],2)
    @assert D == 2

    return Body170{T,D}(
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

function variable_dimension(body::Body170{T,D}) where {T,D}
    if D == 2
        nv = 3 # velocity
        nx = nv
    else
        error("no 3D yet")
    end
    return nx
end

equality_dimension(body) = variable_dimension(body)

function parameter_dimension(body::Body170{T,D}) where {T,D}
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

function get_parameters(body::Body170{T,D}) where {T,D}
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

function set_parameters!(body::Body170{T,D}, θ) where {T,D}
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

function unpack_body_parameters(θ::Vector, body::Body170{T,D}) where {T,D}
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

function body_residual!(e, x, θ, body::Body170)
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


################################################################################
# contact
################################################################################
struct Contact170{T,D,NP,NC}
    name::Symbol
    node_index::NodeIndices170
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function Contact170(A_parent_collider::Matrix{T},
        b_parent_collider::Vector{T},
        A_child_collider::Matrix{T},
        b_child_collider::Vector{T};
        name::Symbol=:contact,
        node_index::NodeIndices170=NodeIndices170()) where {T}

    d = size(A_parent_collider, 2)
    np = size(A_parent_collider, 1)
    nc = size(A_child_collider, 1)
    return Contact170{T,d,np,nc}(
        name,
        node_index,
        A_parent_collider,
        b_parent_collider,
        A_child_collider,
        b_child_collider,
    )
end

function Contact170(parent_body::Body170, child_body::Body170) where {T}
    return Contact170(
        parent_body.A_colliders[1],
        parent_body.b_colliders[1],
        child_body.A_colliders[1],
        child_body.b_colliders[1],
    )
end

function variable_dimension(contact::Contact170{T,D}) where {T,D}
    if D == 2
        nγ = 2*1 # impact (dual and slack)
        # nb = 2*2 # friction (dual and slack)
        nx = nγ# + nb
    else
        error("no 3D yet")
    end
    return nx
end

function equality_dimension(contact::Contact170{T,D}) where {T,D}
    if D == 2
        nγ = 1 # impact slackness equality constraint
        # nb = 2 # friction slackness equality constraints
        nx = nγ# + nb
    else
        error("no 3D yet")
    end
    return nx
end

function parameter_dimension(contact::Contact170{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = nAp + nbp + nAc + nbc
    return nθ
end

function subparameter_dimension(contact::Contact170{T,D,NP,NC}) where {T,D,NP,NC}
    if D == 2
        nx = D+1
        # x_parent, x_child, Ap, bp, Ac, bc
        nθl = nx + nx + NP * (D+1) + NC * (D+1)
    else
        error("no 3D yet")
    end
    return nθl
end

function subvariable_dimension(contact::Contact170{T,D,NP,NC}) where {T,D,NP,NC}
    if D == 2
        nx = D+1
        nθl = subparameter_dimension(contact)
        # ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child
        nxl = 1 + D + D + 1*2nx + D*nθl + D*nθl
    else
        error("no 3D yet")
    end
    return nxl
end

function unpack_contact_variables(x::Vector{T}) where T
    off = 0
    γ = x[off .+ (1:1)]; off += 1
    sγ = x[off .+ (1:1)]; off += 1
    return γ, sγ
end

function get_parameters(contact::Contact170{T,D}) where {T,D}
    θ = [
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::Contact170{T,D,NP,NC}, θ) where {T,D,NP,NC}
    off = 0
    contact.A_parent_collider .= reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    contact.b_parent_collider .= θ[off .+ (1:NP)]; off += NP
    contact.A_child_collider .= reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    contact.b_child_collider .= θ[off .+ (1:NC)]; off += NC
    return nothing
end

function unpack_contact_parameters(θ::Vector, contact::Contact170{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    A_parent_collider = reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    b_parent_collider = θ[off .+ (1:NP)]; off += NP
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return A_parent_collider, b_parent_collider, A_child_collider, b_child_collider
end

function unpack_contact_subvariables(xl::Vector, contact::Contact170{T,D,NP,NC}) where {T,D,NP,NC}
    nθl = subparameter_dimension(contact)

    off = 0
    ϕ = xl[off .+ (1:1)]; off += 1
    p_parent = xl[off .+ (1:D)]; off += D
    p_child = xl[off .+ (1:D)]; off += D
    N = reshape(xl[off .+ (1:2D+2)], (1,2D+2)); off += 2D+2
    ∂p_parent = reshape(xl[off .+ (1:D*nθl)], (D,nθl)); off += D*nθl
    ∂p_child = reshape(xl[off .+ (1:D*nθl)], (D,nθl)); off += D*nθl
    return ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child
end

function contact_residual!(e, x, xl, θ, contact::Contact170, pbody::Body170, cbody::Body170)
    # variables
    γ, sγ = unpack_contact_variables(x[contact.node_index.x])
    # subvariables
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl, contact)

    # dynamics
    e[contact.node_index.e] .+= sγ - ϕ
    e[[pbody.node_index.e; cbody.node_index.e]] .+= -N'*γ
    return nothing
end

################################################################################
# dimensions
################################################################################
struct MechanismDimensions170
    body_configuration::Int
    body_velocity::Int
    body_state::Int
    bodies::Int
    contacts::Int
    variables::Int
    parameters::Int
    primals::Int
    cone::Int
    equality::Int
end

function MechanismDimensions170(bodies::Vector, contacts::Vector)
    # dimensions
    nq = 3 # in 2D
    nv = 3 # in 2D
    nx = 6 # in 2D
    nb = length(bodies)
    nc = length(contacts)
    nx = sum(variable_dimension.(bodies)) + sum(variable_dimension.(contacts))
    nθ = sum(parameter_dimension.(bodies)) + sum(parameter_dimension.(contacts))
    num_primals = sum(variable_dimension.(bodies))
    num_cone = Int(sum(variable_dimension.(contacts)) / 2)
    num_equality = num_primals + num_cone
    return MechanismDimensions170(nq, nv, nx, nb, nc, nx, nθ, num_primals, num_cone, num_equality)
end

################################################################################
# mechanism
################################################################################
struct Mechanism170{T,D,NB,NC}
    variables::Vector{T}
    parameters::Vector{T}
    solver::Solver{T}
    bodies::Vector{Body170{T}}
    contacts::Vector{Contact170{T}}
    dimensions::MechanismDimensions170
    # equalities::Vector{Equality{T}}
    # inequalities::Vector{Inequality{T}}
end

function Mechanism170(bodies::Vector, contacts::Vector;
        options::Options{T}=Options(), D::Int=2) where {T}

    # Dimensions
    dim = MechanismDimensions170(bodies, contacts)

    # indexing
    indexing!([bodies; contacts])

    # solver
    parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)

    methods = mechanism_methods(bodies, contacts, dim)
    solver = Solver(
            nothing,
            dim.primals,
            dim.cone,
            parameters=parameters,
            nonnegative_indices=collect(1:dim.cone),
            second_order_indices=[collect(1:0)],
            methods=methods,
            options=options
            )

    # vectors
    variables = solver.solution.all
    parameters = solver.parameters

    nb = length(bodies)
    nc = length(contacts)
    mechanism = Mechanism170{T,D,nb,nc}(
        variables,
        parameters,
        solver,
        bodies,
        contacts,
        dim,
        )
    return mechanism
end

function indexing!(nodes::Vector)
    eoff = 0
    xoff = 0
    θoff = 0
    for node in nodes
        ne = equality_dimension(node)
        nx = variable_dimension(node)
        nθ = parameter_dimension(node)
        node.node_index.e = collect(eoff .+ (1:ne)); eoff += ne
        node.node_index.x = collect(xoff .+ (1:nx)); xoff += nx
        node.node_index.θ = collect(θoff .+ (1:nθ)); θoff += nθ
    end
    return nothing
end

################################################################################
# methods
################################################################################
function generate_gradients(func::Function, num_equality::Int, num_variables::Int,
        num_parameters::Int;
        checkbounds=true,
        threads=false)

    f = Symbolics.variables(:f, 1:num_equality)
    e = Symbolics.variables(:e, 1:num_equality)
    x = Symbolics.variables(:x, 1:num_variables)
    θ = Symbolics.variables(:θ, 1:num_parameters)

    f .= e
    func(f, x, θ)

    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, e, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fx_expr, fθ_expr, fx_sparsity, fθ_sparsity
end

abstract type NodeMethods170{T,E,EX,Eθ} end

struct DynamicsMethods170{T} <: AbstractProblemMethods{T}
    methods::Vector{NodeMethods170}
    α::T
end

struct BodyMethods170{T,E,EX,Eθ} <: NodeMethods170{T,E,EX,Eθ}
    equality_constraint::E
    equality_jacobian_variables::EX
    equality_jacobian_parameters::Eθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function BodyMethods170(body::Body170, dimensions::MechanismDimensions170)
    r!(e, x, θ) = body_residual!(e, x, θ, body)
    f, fx, fθ, fx_sparsity, fθ_sparsity = generate_gradients(r!, dimensions.equality,
        dimensions.variables, dimensions.parameters)
    return BodyMethods170(
        f,
        fx,
        fθ,
        zeros(length(fx_sparsity)),
        zeros(length(fθ_sparsity)),
        fx_sparsity,
        fθ_sparsity,
        )
end

struct ContactMethods170{T,E,EX,Eθ,C,S} <: NodeMethods170{T,E,EX,Eθ}
    contact_solver::C
    subvariables::Vector{T}
    subparameters::Vector{T}

    set_subparameters!::S
    equality_constraint::E
    equality_jacobian_variables::EX
    equality_jacobian_parameters::Eθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function ContactMethods170(contact::Contact170, pbody::Body170, cbody::Body170,
        dimensions::MechanismDimensions170;
        checkbounds=true,
        threads=false)


    contact_solver = ContactSolver(
        contact.A_parent_collider,
        contact.b_parent_collider,
        contact.A_child_collider,
        contact.b_child_collider,
        )

    num_equality = dimensions.equality
    num_variables = dimensions.variables
    num_parameters = dimensions.parameters
    num_subvariables = contact_solver.num_subvariables
    num_subparameters = contact_solver.num_subparameters
    subvariables = zeros(num_subvariables)
    subparameters = zeros(num_subparameters)

    # set_subparameters!
    x = Symbolics.variables(:x, 1:num_variables)
    θ = Symbolics.variables(:θ, 1:num_parameters)
    v25_parent = unpack_body_variables(x[pbody.node_index.x])
    v25_child = unpack_body_variables(x[cbody.node_index.x])

    x2_parent, _, _, timestep_parent = unpack_body_parameters(θ[pbody.node_index.θ])
    x2_child, _, _, timestep_child = unpack_body_parameters(θ[cbody.node_index.θ])
    x3_parent = x2_parent .+ timestep_parent[1] * v25_parent
    x3_child = x2_child .+ timestep_child[1] * v25_child
    @show x2_parent, timestep_parent, x3_parent, v25_parent

    Ap, bp, Ac, bc = unpack_contact_parameters(θ[contact.node_index.θ], contact)

    # θl = fct(x, θ)
    θl = [x3_parent; x3_child; vec(Ap); bp; vec(Ac); bc]

    set_subparameters! = Symbolics.build_function(θl, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    # evaluation
    f = Symbolics.variables(:f, 1:num_equality)
    e = Symbolics.variables(:e, 1:num_equality)
    xl = Symbolics.variables(:xl, 1:num_subvariables)
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl, contact)


    f .= e
    contact_residual!(f, x, xl, θ, contact, pbody, cbody)

    # for this one we are missing only third order tensors
    fx = Symbolics.sparsejacobian(f, x)
    fx[contact.node_index.e, [pbody.node_index.x; cbody.node_index.x]] .+= -sparse(N)
    # for this one we are missing ∂ϕ/∂θ and third order tensors
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, e, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, xl, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return ContactMethods170(
        contact_solver,
        subvariables,
        subparameters,
        set_subparameters!,
        f_expr,
        fx_expr,
        fθ_expr,
        zeros(length(fx_sparsity)),
        zeros(length(fθ_sparsity)),
        fx_sparsity,
        fθ_sparsity,
    )
end

function mechanism_methods(bodies::Vector, contacts::Vector, dimensions::MechanismDimensions170)
    methods = Vector{NodeMethods170}()

    # body
    for body in bodies
        push!(methods, BodyMethods170(body, dimensions))
    end

    # contact
    for contact in contacts
        # TODO here we need to avoid hardcoding body1 and body2 as paretn and child
        push!(methods, ContactMethods170(contact, bodies[1], bodies[2], dimensions))
    end

    return DynamicsMethods170(methods, 1.0)
end

################################################################################
# evaluate
################################################################################

# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::Vector{NodeMethods170}) where T
#     e .= 0.0
#     ex .= 0.0
#     eθ .= 0.0
#     for m in methods
#         evaluate!(e, ex, eθ, x, θ, m)
#     end
# end
#
# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::BodyMethods170{T,E,EX,Eθ}) where {T,E,EX,Eθ}
#
#     methods.equality_constraint(e, e, x, θ)
#     methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
#     methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
#
#     for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
#         ex[idx...] += methods.equality_jacobian_variables_cache[i]
#     end
#     for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
#         eθ[idx...] += methods.equality_jacobian_parameters_cache[i]
#     end
# end
#
# function evaluate!(e::Vector{T}, ex::Matrix{T}, eθ::Matrix{T},
#         x::Vector{T}, θ::Vector{T}, methods::ContactMethods170{T,S}) where {T,S}
#
#     contact_solver = methods.contact_solver
#     xl = methods.subvariables
#     θl = methods.subparameters
#
#     # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
#     methods.set_subparameters!(θl, x, θ)
#     update_subvariables!(xl, θl, contact_solver)
#
#     # modify e, ex, eθ in-place using symbolics methods taking x, θ, xl as inputs
#     methods.equality_constraint(e, e, x, xl, θ)
#     methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, xl, θ)
#     methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, xl, θ)
#
#     for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
#         ex[idx...] += methods.equality_jacobian_variables_cache[i]
#     end
#     for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
#         eθ[idx...] += methods.equality_jacobian_parameters_cache[i]
#     end
# end

function evaluate!(
        problem::ProblemData{T},
        # methods::Vector{NodeMethods170},
        methods::DynamicsMethods170{T},
        cone_methods::ConeMethods{B,BX,P,PX,PXI,TA},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        cone_jacobian_inverse=false,
        ) where {T,B,BX,P,PX,PXI,TA}

    # TODO this method allocates memory, need fix

    # reset
    problem.equality_constraint .= 0.0
    problem.equality_jacobian_variables .= 0.0
    problem.equality_jacobian_parameters .= 0.0

    # apply all methods
    for method in methods.methods
        evaluate!(problem, method, solution, parameters;
            equality_constraint=equality_constraint,
            equality_jacobian_variables=equality_jacobian_variables,
            equality_jacobian_parameters=equality_jacobian_parameters)
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
        cone_jacobian_inverse=cone_jacobian_inverse,
        cone_target=true # TODO this should only be true once at the beginning of the solve
    )

    return nothing
end

function evaluate!(problem::ProblemData{T},
        methods::BodyMethods170{T,E,EX,Eθ},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        ) where {T,E,EX,Eθ}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    (equality_constraint && ne > 0) && methods.equality_constraint(
        problem.equality_constraint, problem.equality_constraint, x, θ)

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] += methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
            problem.equality_jacobian_parameters[idx...] += methods.equality_jacobian_parameters_cache[i]
        end
    end
    return
end

function evaluate!(problem::ProblemData{T},
        # methods::ContactMethods170{T,E,EX,Eθ},
        methods::ContactMethods170{T,S},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        # ) where {T,E,EX,Eθ}
        ) where {T,S}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
    contact_solver = methods.contact_solver
    xl = methods.subvariables
    θl = methods.subparameters
    methods.set_subparameters!(θl, x, θ)
    update_subvariables!(xl, θl, contact_solver)
    # @show x[1:3]
    # @show x[4:6]
    # @show θl[1:3]
    # @show θl[4:6]
    # @show xl[1]

    # update equality constraint and its jacobiens
    (equality_constraint && ne > 0) && methods.equality_constraint(
        problem.equality_constraint, problem.equality_constraint, x, xl, θ)

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, xl, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] += methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, xl, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
            problem.equality_jacobian_parameters[idx...] += methods.equality_jacobian_parameters_cache[i]
        end
    end
    return
end


################################################################################
# demo
################################################################################
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
bp = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bc = 0.5*[
     1,
     1,
     1,
     1,
    ]


timestep = 0.01
gravity = -0.0*9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
bodya = Body170(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body170(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact170(bodies[1], bodies[2])]

dim = MechanismDimensions170(bodies, contacts)
indexing!([bodies; contacts])

θbody = get_parameters(bodya)
θcontact = get_parameters(contacts[1])
set_parameters!(bodya, θbody)
set_parameters!(contacts[1], θcontact)

x0 = rand(dim.variables)
θ0 = rand(dim.parameters)
e0 = zeros(dim.variables)
ex0 = zeros(dim.variables, dim.variables)
eθ0 = zeros(dim.variables, dim.parameters)

contact_solver = ContactSolver(Ap, bp, Ac, bc)
contact_methods = ContactMethods170(contacts[1], bodies..., dim)

problem_methods0 = mechanism_methods(bodies, contacts, dim)
methods0 = problem_methods0.methods
# evaluate!(e0, ex0, eθ0, x0, θ0, methods0)
# Main.@profiler [evaluate!(e0, ex0, eθ0, x0, θ0, methods0) for i=1:5000]
# @benchmark $evaluate!($e0, $ex0, $eθ0, $x0, $θ0, $methods0)




dim_solver = Dimensions(dim.primals, dim.cone, dim.parameters)
index_solver = Indices(dim.primals, dim.cone, dim.parameters)
problem = ProblemData(dim.variables, dim.parameters, dim.equality, dim.cone)
idx_nn = collect(1:dim.cone)
idx_soc = [collect(1:0)]
cone_methods = ConeMethods(dim.cone, idx_nn, idx_soc)
solution = Point(dim_solver, index_solver)
solution.all .= 1.0
parameters = ones(dim.parameters)

body_method0 = methods0[1]
contact_method0 = methods0[3]

evaluate!(
        problem,
        problem_methods0,
        cone_methods,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        )
Main.@code_warntype evaluate!(
        problem,
        problem_methods0,
        cone_methods,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        )
@benchmark $evaluate!(
        $problem,
        $problem_methods0,
        $cone_methods,
        $solution,
        $parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        )

evaluate!(problem,
        contact_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
Main.@code_warntype evaluate!(problem,
        contact_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
@benchmark $evaluate!($problem,
        $contact_method0,
        $solution,
        $parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)

evaluate!(problem,
        body_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
Main.@code_warntype evaluate!(problem,
        body_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
@benchmark $evaluate!($problem,
        $body_method0,
        $solution,
        $parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)












################################################################################
# test mechanism
################################################################################
mech = Mechanism170(bodies, contacts)
solver = mech.solver

bodies[1].pose .= [+10,0,0.0]
bodies[2].pose .= [-10,0,0.0]

θb1 = get_parameters(bodies[1])
θb2 = get_parameters(bodies[2])
θc1 = get_parameters(contacts[1])
parameters = [θb1; θb2; θc1]

solver.parameters .= parameters
evaluate!(solver.problem,
    solver.methods,
    solver.cone_methods,
    solver.solution,
    solver.parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    equality_jacobian_parameters=true,
    cone_constraint=true,
    cone_jacobian=true,
    cone_jacobian_inverse=true,
    )

function residual(variables, parameters, bodies, contacts, contact_methods)
    num_equality = 7
    num_variables = length(variables)

    x = variables
    θ = parameters

    e = zeros(num_equality)
    for body in bodies
        body_residual!(e, x, θ, body)
    end
    for contact in contacts
        # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
        contact_solver = contact_methods[1].contact_solver
        xl = contact_methods[1].subvariables
        θl = contact_methods[1].subparameters
        contact_methods[1].set_subparameters!(θl, x, θ)
        update_subvariables!(xl, θl, contact_solver)
        contact_residual!(e, x, xl, θ, contact, bodies[1], bodies[2])
    end
    return e
end


solver.solution.all .= [+0,0,0, -0,0,0, 1e-7, 10.5]
evaluate!(solver.problem,
    solver.methods,
    solver.cone_methods,
    solver.solution,
    solver.parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    equality_jacobian_parameters=true,
    cone_constraint=true,
    cone_jacobian=true,
    cone_jacobian_inverse=true,
    )

e0 = residual(solver.solution.all, solver.parameters, bodies, contacts, methods0[3:3])
e1 = deepcopy(solver.problem.equality_constraint)
e0
e1
norm(e0 - e1, Inf)

J0 = FiniteDiff.finite_difference_jacobian(
    variables -> residual(variables, solver.parameters, mech.bodies, mech.contacts,
        methods0[3:3]), mech.solver.solution.all)
J1 = deepcopy(mech.solver.problem.equality_jacobian_variables)
norm(J0 - J1, Inf)

mech.bodies[1].pose
mech.bodies[2].pose


using Plots
plot(Gray.(1e3abs.(J0)))
plot(Gray.(1e3abs.(J1)))
plot(Gray.(1e3abs.(J1 - J0)))

J0[3,:]
J1[3,:]
J0[6,:]
J1[6,:]


# function finite_diff_jacobian(solver)
#     evaluate!(solver.problem,
#         solver.methods,
#         solver.cone_methods,
#         solver.solution,
#         solver.parameters,
#         equality_constraint=true,
#         equality_jacobian_variables=true,
#         equality_jacobian_parameters=true,
#         cone_constraint=true,
#         cone_jacobian=true,
#         cone_jacobian_inverse=true,
#         )
#
#
#     return

parameter_dimension(bodies[1])
parameter_dimension(contacts[1])
mech.solver.options.verbose = false
@benchmark $solve!($mech.solver)
Main.@profiler [solve!(mech.solver) for i=1:1000]





################################################################################
# test simulation
################################################################################

mech = Mechanism170(bodies, contacts)
mech.solver.methods.methods[3].contact_solver.solver.options.complementarity_tolerance=1e-2
mech.solver.methods.methods[3].contact_solver.solver.options.residual_tolerance=1e-10
# mech.solver.methods.methods[3].contact_solver.solver.options.verbose=true
Xa2 = [[+1,1,0.0]]
Xb2 = [[-1,1,0.0]]
Va15 = [[-2,0,0.0]]
Vb15 = [[+0,0,0.0]]
Pp = []
Pc = []
iter = []

H = 60
for i = 1:H
    mech.bodies[1].pose .= Xa2[end]
    mech.bodies[1].velocity .= Va15[end]
    mech.bodies[2].pose .= Xb2[end]
    mech.bodies[2].velocity .= Vb15[end]

    θb1 = get_parameters(mech.bodies[1])
    θb2 = get_parameters(mech.bodies[2])
    θc1 = get_parameters(mech.contacts[1])
    mech.parameters .= [θb1; θb2; θc1]
    mech.solver.parameters .= [θb1; θb2; θc1]

    # mech = Mechanism170(bodies, contacts)
    solve!(mech.solver)
    va25 = deepcopy(mech.solver.solution.all[1:3])
    vb25 = deepcopy(mech.solver.solution.all[4:6])
    push!(Va15, va25)
    push!(Vb15, vb25)
    push!(Xa2, Xa2[end] + timestep * va25)
    push!(Xb2, Xb2[end] + timestep * vb25)
    p_parent = deepcopy(mech.solver.methods.methods[3].subvariables[2:3]) + (Xa2[end][1:2] + Xb2[end][1:2])/2
    p_child = deepcopy(mech.solver.methods.methods[3].subvariables[4:5]) + (Xa2[end][1:2] + Xb2[end][1:2])/2
    push!(Pp, p_parent)
    push!(Pc, p_child)
    push!(iter, mech.solver.trace.iterations)
end

scatter(iter)

################################################################################
# visualization
################################################################################
set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polyhedron!(vis, Ap, bp, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polyhedron!(vis, Ac, bc, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_2d_polyhedron!(vis, Xa2[i][1:2], Xa2[i][3:3], name=:polya)
        set_2d_polyhedron!(vis, Xb2[i][1:2], Xb2[i][3:3], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Pp[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
render(vis)
# open(vis)
# convert_frames_to_video_and_gif("polytope_thrashing_collision")

# n = 2
# m = 5
# @variables a[1:n]
# @variables r[1:m]
# a = Symbolics.scalarize(a)
# r = Symbolics.scalarize(r)
#
# out = 1.0*r
# out[2 .+ (1:n)] .+= a
# r
# expr = build_function(out, r, a)[2]
# ftest = eval(expr)
#
# out0 = zeros(m)
# r0 = zeros(m)
# a0 = ones(n)
# ftest(out0, r0, a0)
# out0
# @benchmark $ftest($out0, $r0, $a0)


#
# num_variables = dim.variables
# num_parameters = dim.parameters
# @variables out[1:num_variables]
# @variables r[1:num_variables]
# @variables x[1:num_variables]
# @variables θ[1:num_parameters]
# out = Symbolics.scalarize(out)
# r = Symbolics.scalarize(r)
# x = Symbolics.scalarize(x)
# θ = Symbolics.scalarize(θ)
#
# indexing!([bodies; contacts])
# out .= r
# residuals[1](out, x, θ)
# r
# out
# symbolic_residual = eval(build_function(out, r, x, θ)[2])
#
# r0 = ones(num_variables)
# x0 = zeros(num_variables)
# θ0 = ones(num_parameters)
# symbolic_residual(r0, r0, x0, θ0)
# r0
# @benchmark $symbolic_residual($r0, $r0, $x0, $θ0)
#
#
# generate_residual(residuals[1], num_variables, num_parameters)
# generate_residual(residuals[2], num_variables, num_parameters)
