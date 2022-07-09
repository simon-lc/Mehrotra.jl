################################################################################
# dimensions
################################################################################
struct MechanismDimensions171
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

function MechanismDimensions171(bodies::Vector, contacts::Vector)
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
    return MechanismDimensions171(nq, nv, nx, nb, nc, nx, nθ, num_primals, num_cone, num_equality)
end

################################################################################
# mechanism
################################################################################
struct Mechanism171{T,D,NB,NC,C}
    variables::Vector{T}
    parameters::Vector{T}
    solver::Solver{T}
    bodies::Vector{Body171{T}}
    contacts::Vector{C}
    dimensions::MechanismDimensions171
    # equalities::Vector{Equality{T}}
    # inequalities::Vector{Inequality{T}}
end

function Mechanism171(residual, bodies::Vector, contacts::Vector;
        options::Options{T}=Options(), D::Int=2) where {T}

    # Dimensions
    dim = MechanismDimensions171(bodies, contacts)

    # indexing
    indexing!([bodies; contacts])

    # solver
    parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)

    # methods = mechanism_methods(bodies, contacts, dim)
    solver = Solver(
            residual,
            dim.primals,
            dim.cone,
            parameters=parameters,
            nonnegative_indices=collect(1:dim.cone),
            second_order_indices=[collect(1:0)],
            method_type=:finite_difference,
            options=options
            )
    # solver = Solver(
    #         nothing,
    #         dim.primals,
    #         dim.cone,
    #         parameters=parameters,
    #         nonnegative_indices=collect(1:dim.cone),
    #         second_order_indices=[collect(1:0)],
    #         methods=methods,
    #         options=options
    #         )

    # vectors
    variables = solver.solution.all
    parameters = solver.parameters

    nb = length(bodies)
    nc = length(contacts)
    mechanism = Mechanism171{T,D,nb,nc,eltype(contacts)}(
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

function mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)
    num_duals = length(duals)
    num_primals = length(primals)
    num_equality = num_primals + num_duals

    e = zeros(num_equality)
    x = [primals; duals; slacks]
    θ = parameters

    # body
    for body in bodies
        body_residual!(e, x, θ, body)
    end

    # contact
    for contact in contacts
        contact_residual!(e, x, θ, contact, bodies[1], bodies[2])
    end
    return e
end
