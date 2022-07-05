
# function step!(mechanism::Mechanism141{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism141{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism141{T})
# end
#
# function set_input!(mechanism::Mechanism141{T})
# end
#
# function set_current_state!(mechanism::Mechanism141{T})
# end
#
# function set_next_state!(mechanism::Mechanism141{T})
# end
#
# function get_current_state!(mechanism::Mechanism141{T})
# end
#
# function get_next_state!(mechanism::Mechanism141{T})
# end

mutable struct NodeIndices141
    e::Vector{Int}
    x::Vector{Int}
    θ::Vector{Int}
end

function NodeIndices141()
    return NodeIndices141(
        collect(1:0),
        collect(1:0),
        collect(1:0),
    )
end

################################################################################
# body
################################################################################
struct Body141{T,D}
    name::Symbol
    node_index::NodeIndices141
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

function Body141(timestep, mass, inertia::Matrix,
        A_colliders::Vector{Matrix{T}},
        b_colliders::Vector{Vector{T}};
        gravity=-9.81,
        name::Symbol=:body,
        node_index::NodeIndices141=NodeIndices141()) where T

    D = size(A_colliders[1])[2]
    @assert D == 2

    return Body141{T,D}(
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

function variable_dimension(body::Body141{T,D}) where {T,D}
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

function parameter_dimension(body::Body141{T,D}) where {T,D}
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

function get_parameters(body::Body141{T,D}) where {T,D}
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

function set_parameters!(body::Body141{T,D}, θ) where {T,D}
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

function body_residual!(r, x, θ, node_index)
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
    dyn = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    r[node_index.r] .+= dyn
    return nothing
end



################################################################################
# contact
################################################################################
struct Contact141{T,D,NP,NC}
    name::Symbol
    node_index::NodeIndices141
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function Contact141(A_parent_collider::Matrix{T},
        b_parent_collider::Vector{T},
        A_child_collider::Matrix{T},
        b_child_collider::Vector{T};
        name::Symbol=:contact,
        node_index::NodeIndices141=NodeIndices141(),) where {T}
    D = size(A_parent_collider)[2]
    NP = size(A_parent_collider)[1]
    NC = size(A_child_collider)[1]
    return Contact141{T,D,NP,NC}(
        name,
        node_index,
        A_parent_collider,
        b_parent_collider,
        A_child_collider,
        b_child_collider,
    )
end

function Contact141(parent_body::Body141, child_body::Body141) where {T}
    return Contact141(
        parent_body.A_colliders[1],
        parent_body.b_colliders[1],
        child_body.A_colliders[1],
        child_body.b_colliders[1],
    )
end

function variable_dimension(contact::Contact141{T,D}) where {T,D}
    if D == 2
        nγ = 2*1 # impact (dual and slack)
        nb = 2*2 # friction (dual and slack)
        nx = nγ + nb
    else
        error("no 3D yet")
    end
    return nx
end

function parameter_dimension(contact::Contact141{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = nAp + nbp + nAc + nbc
    return nθ
end

function get_parameters(contact::Contact141{T,D}) where {T,D}
    θ = [
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::Contact141{T,D,NP,NC}, θ) where {T,D,NP,NC}
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
struct MechanismDimensions141
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
struct Mechanism141{T,D,NB,NC}
    variables::Vector{T}
    parameters::Vector{T}
    solver::Solver228{T}
    bodies::Vector{Body141{T}}
    contacts::Vector{Contact141{T}}
    dimensions::MechanismDimensions141
    # equalities::Vector{Equality{T}}
    # inequalities::Vector{Inequality{T}}
end

function Mechanism141(bodies::Vector, contacts::Vector;
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
    dim = MechanismDimensions141(nq, nv, nx, nb, nc, nx, nθ, num_primals, num_cone)

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
            nonnegative_indices=collect(1:num_cone),
            second_order_indices=[collect(1:0)],
            methods=methods,
            options=options
            )

    # vectors
    variables = solver.solution.all
    parameters = solver.parameters

    mechanism = Mechanism141{T,D,nb,nc}(
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
        ne = variable_dimension(node)
        nx = variable_dimension(node)
        nθ = parameter_dimension(node)
        node.node_index.e = collect(eoff .+ (1:ne)); roff += ne
        node.node_index.x = collect(xoff .+ (1:nx)); xoff += nx
        node.node_index.θ = collect(θoff .+ (1:nθ)); θoff += nθ
    end

    return nothing
end

################################################################################
# demo
################################################################################
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
bodya = Body141(timestep, mass, inertia, [Aa], [ba], gravity=gravity, name=:bodya)
bodyb = Body141(timestep, mass, inertia, [Ab], [bb], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact141(bodies[1], bodies[2])]
indexing!([bodies; contacts])


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
dim = MechanismDimensions141(nq, nv, nx, nb, nc, nx, nθ, num_primals, num_cone)

function generate_residual(func::Function, num_variables::Int, num_parameters::Int;
        checkbounds=true,
        threads=false)

    f = Symbolics.variables(:f, 1:num_variables)
    e = Symbolics.variables(:e, 1:num_variables)
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

function mechanism_methods(bodies::Vector, contacts::Vector, dim::MechanismDimensions141)

    residuals = [r!(r, x, θ) = body_residual!(r, x, θ, body.node_index) for body in bodies]
    methods = Vector{NodeMethods141}()
    for residual in residuals
        f, fx, fθ, fx_sparsity, fθ_sparsity = generate_residual(residual, dim.variables, dim.parameters)
        m = NodeMethods141(
            f,
            fx,
            fθ,
            zeros(length(fx_sparsity)),
            zeros(length(fθ_sparsity)),
            fx_sparsity,
            fθ_sparsity,
            )
        push!(methods, m)
    end

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

    return methods
end

struct NodeMethods141{T,E,EX,Eθ}
    eqaulity::R
    eqaulity_jacobian_variables::RX
    eqaulity_jacobian_parameters::Rθ
    eqaulity_jacobian_variables_cache::Vector{T}
    eqaulity_jacobian_parameters_cache::Vector{T}
    eqaulity_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    eqaulity_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end


methods0 = mechanism_methods(bodies, contacts, dim)
function evaluate!(r::Vector{T}, x::Vector{T}, θ::Vector{T}, methods::Vector{NodeMethods141}) where T
    for m in methods
        evaluate!(r, x, θ, m)
    end
end

function evaluate!(equality_constraint::Vector{T}, equality_jacobian_variables::Matrix{T},
    equality_jacobian_parameters::Matrix{T}, x::Vector{T}, θ::Vector{T}, methods::NodeMethods141{T,E,EX,Eθ}) where {T,E,EX,Eθ}

    methods.residual(equality_constraint, equality_constraint, x, θ)
    methods.residual_jacobian_variables(methods.residual_jacobian_variables_cache, x, θ)
    methods.residual_jacobian_parameters(methods.residual_jacobian_parameters_cache, x, θ)
end



methods0
evaluate!(r0, x0, θ0, methods0)
@benchmark $evaluate!($r0, $x0, $θ0, $methods0)



methods0


# solver = mechanism_solver(bodies, contacts, dim)
# mech = Mechanism141(bodies, contacts)















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
