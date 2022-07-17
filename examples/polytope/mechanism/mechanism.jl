################################################################################
# dimensions
################################################################################
struct MechanismDimensions182
    body_configuration::Int
    body_velocity::Int
    body_state::Int
    body_input::Int
    state::Int
    input::Int
    bodies::Int
    contacts::Int
    variables::Int
    parameters::Int
    # primals::Int
    # duals::Int
    # slacks::Int
    # cone::Int
    # equality::Int
end

function MechanismDimensions182(bodies::Vector, contacts::Vector)
    # dimensions
    body_configuration = 3 # in 2D
    body_velocity = 3 # in 2D
    body_state = 6 # in 2D
    body_input = 3 # in 2D

    num_bodies = length(bodies)
    num_contacts = length(contacts)


    state = num_bodies * body_state
    input = num_bodies * body_input

    nodes = [bodies; contacts]
    num_variables = sum(variable_dimension.(nodes))
    num_parameters = sum(parameter_dimension.(nodes))
    # num_primals = sum(primal_dimension.(nodes))
    # num_cone = sum(cone_dimension.(nodes))
    # num_duals = num_cone
    # num_slacks = num_cone
    # num_equality = sum(equality_dimension.(nodes))

    return MechanismDimensions182(
        body_configuration,
        body_velocity,
        body_state,
        body_input,
        state, 
        input,
        num_bodies,
        num_contacts,
        num_variables,
        num_parameters,
        # num_primals,
        # num_duals,
        # num_slacks,
        # num_cone,
        # num_equality
        )
end

################################################################################
# mechanism
################################################################################
struct Mechanism182{T,D,NB,NC,C}
    variables::Vector{T}
    parameters::Vector{T}
    solver::Solver{T}
    bodies::Vector{Body182{T}}
    contacts::Vector{C}
    dimensions::MechanismDimensions182
    # equalities::Vector{Equality{T}}
    # inequalities::Vector{Inequality{T}}
end

function Mechanism182(residual, bodies::Vector, contacts::Vector;
        options::Options{T}=Options(), D::Int=2) where {T}

    # # Dimensions
    nodes = [bodies; contacts]
    dim = MechanismDimensions182(bodies, contacts)
    num_primals = sum(primal_dimension.(nodes))
    num_cone = sum(cone_dimension.(nodes))

    # indexing
    indexing!(nodes)

    # solver
    parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)

    # methods = mechanism_methods(bodies, contacts, dim)
    solver = Solver(
            residual,
            num_primals,
            num_cone,
            parameters=parameters,
            nonnegative_indices=collect(1:num_cone),
            second_order_indices=[collect(1:0)],
            method_type=:finite_difference,
            options=options
            )

    # vectors
    variables = solver.solution.all
    parameters = solver.parameters

    nb = length(bodies)
    nc = length(contacts)
    mechanism = Mechanism182{T,D,nb,nc,eltype(contacts)}(
        variables,
        parameters,
        solver,
        bodies,
        contacts,
        dim,
        )
    return mechanism
end

function mechanism_residual(primals, duals, slacks, parameters, 
        bodies::Vector, contacts::Vector)

    num_duals = length(duals)
    num_primals = length(primals)
    num_equality = num_primals + num_duals

    e = zeros(num_equality)
    x = [primals; duals; slacks]
    θ = parameters

    # body
    for body in bodies
        residual!(e, x, θ, body)
    end

    # contact
    for contact in contacts
        residual!(e, x, θ, contact, bodies)
    end
    return e
end

function residual!(e, x, θ, contact::PolyPoly182, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    cbody = find_body(bodies, contact.child_name)
    residual!(e, x, θ, contact, pbody, cbody)
    return nothing
end

function residual!(e, x, θ, contact::PolyHalfSpace182, bodies::Vector)
    pbody = find_body(bodies, contact.parent_name)
    residual!(e, x, θ, contact, pbody)
    return nothing
end

