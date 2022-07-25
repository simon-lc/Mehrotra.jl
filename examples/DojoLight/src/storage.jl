
mutable struct Storage116{T,H}
    z::Vector{Vector{T}} # H x nz
    u::Vector{Vector{T}} # H x nu
    x::Vector{Vector{Vector{T}}} # H x nb x nx
    v::Vector{Vector{Vector{T}}} # H x nb x nv
    contact_point::Vector{Vector{Vector{T}}} # H x nc x d
    normal::Vector{Vector{Vector{T}}} # H x nc x d
    tangent::Vector{Vector{Vector{T}}} # H x nc x d
    variables::Vector{Vector{T}} # H x variables
    iterations::Vector{Int} # H
end

function Storage(dim::MechanismDimensions183, H::Int, T=Float64)
    z = [zeros(T, dim.state) for i = 1:H]
    u = [zeros(T, dim.input) for i = 1:H]
    x = [[zeros(T, dim.body_configuration) for j = 1:dim.bodies] for i = 1:H]
    v = [[zeros(T, dim.body_velocity) for j = 1:dim.bodies] for i = 1:H]
    contact_point = [[zeros(T, 2) for j = 1:dim.contacts] for i = 1:H]
    normal = [[zeros(T, 2) for j = 1:dim.contacts] for i = 1:H]
    tangent = [[zeros(T, 2) for j = 1:dim.contacts] for i = 1:H]
    variables = [zeros(Int, dim.variables) for i = 1:H]
    iterations = zeros(Int, H)
    storage = Storage116{T,H}(z, u, x, v, contact_point, normal, tangent, variables, iterations)
    return storage
end

function record!(storage::Storage116{T,H}, mechanism::Mechanism183{T,D,NB,NC}, i::Int) where {T,H,D,NB,NC}
    storage.z[i] .= get_current_state(mechanism)
    storage.u[i] .= get_input(mechanism)

    for j = 1:NB
        storage.x[i][j] .= mechanism.bodies[j].pose
        storage.v[i][j] .= mechanism.bodies[j].velocity
    end

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters
    for (j, contact) in enumerate(mechanism.contacts)
        contact_point, normal, tangent = contact_frame(contact, mechanism)
        storage.contact_point[i][j] .= contact_point
        storage.normal[i][j] .= normal
        storage.tangent[i][j] .= tangent
    end

    storage.variables[i] .= variables
    storage.iterations[i] = mechanism.solver.trace.iterations

    return nothing
end
