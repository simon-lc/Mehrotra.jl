################################################################################
# dimensions
################################################################################
struct MechanismDimensions175
    body_configuration::Int
    body_velocity::Int
    body_state::Int
    bodies::Int
    contacts::Int
    variables::Int
    parameters::Int
    primals::Int
    duals::Int
    slacks::Int
    cone::Int
    equality::Int
end

function MechanismDimensions175(bodies::Vector, contacts::Vector)
    # dimensions
    body_configuration = 3 # in 2D
    body_velocity = 3 # in 2D
    body_state = 6 # in 2D

    num_bodies = length(bodies)
    num_contacts = length(contacts)

    nodes = [bodies; contacts]
    num_variables = sum(variable_dimension.(nodes))
    num_parameters = sum(parameter_dimension.(nodes))
    num_primals = sum(primal_dimension.(nodes))
    num_cone = sum(cone_dimension.(nodes))
    num_duals = num_cone
    num_slacks = num_cone
    num_equality = sum(equality_dimension.(nodes))

    return MechanismDimensions175(
        body_configuration,
        body_velocity,
        body_state,
        num_bodies,
        num_contacts,
        num_variables,
        num_parameters,
        num_primals,
        num_duals,
        num_slacks,
        num_cone,
        num_equality)
end

################################################################################
# mechanism
################################################################################
struct Mechanism175{T,D,NB,NC,C}
    variables::Vector{T}
    parameters::Vector{T}
    solver::Solver{T}
    bodies::Vector{Body175{T}}
    contacts::Vector{C}
    dimensions::MechanismDimensions175
    # equalities::Vector{Equality{T}}
    # inequalities::Vector{Inequality{T}}
end

function Mechanism175(residual, bodies::Vector, contacts::Vector;
        options::Options{T}=Options(), D::Int=2) where {T}

    # Dimensions
    dim = MechanismDimensions175(bodies, contacts)

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

    # vectors
    variables = solver.solution.all
    parameters = solver.parameters

    nb = length(bodies)
    nc = length(contacts)
    mechanism = Mechanism175{T,D,nb,nc,eltype(contacts)}(
        variables,
        parameters,
        solver,
        bodies,
        contacts,
        dim,
        )
    return mechanism
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
