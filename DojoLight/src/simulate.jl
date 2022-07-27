function update_parameters!(mechanism::Mechanism1170)
    bodies = mechanism.bodies
    contacts = mechanism.contacts
    solver = mechanism.solver

    off = 0
    for node in [bodies; contacts]
        θ = get_parameters(node)
        nθ = parameter_dimension(node)
        solver.parameters[off .+ (1:nθ)] .= θ; off += nθ
    end
    # update the consistency logic
    solver.consistency.solved .= false
    set_bool!(solver.consistency.differentiated, false)
    return nothing
end

function update_nodes!(mechanism::Mechanism1170)
    bodies = mechanism.bodies
    contacts = mechanism.contacts
    solver = mechanism.solver

    off = 0
    for node in [bodies; contacts]
        nθ = parameter_dimension(node)
        θ = solver.parameters[off .+ (1:nθ)]; off += θ
        set_parameters!(node, θ)
    end
    return nothing
end

function set_input!(mechanism::Mechanism1170, u)
    off = 0
    for body in mechanism.bodies
        nu = length(body.input)
        body.input .= u[off .+ (1:nu)]; off += nu
    end
    return nothing
end

function get_input(mechanism::Mechanism1170{T,D,NB}) where {T,D,NB}
    off = 0
    nu = sum(input_dimension.(mechanism.bodies))
    u = zeros(nu)
    for body in mechanism.bodies
        ni = length(body.input)
        u[off .+ (1:ni)] .= body.input; off += ni
    end
    return u
end

function set_current_state!(mechanism::Mechanism1170, z)
    off = 0

    for body in mechanism.bodies
        nx = state_dimension(body)
        set_current_state!(body, view(z, off .+ (1:nx))); off += nx
    end
    return nothing
end

function get_current_state(mechanism::Mechanism1170{T,D,NB}) where {T,D,NB}
    nz = sum(state_dimension.(mechanism.bodies))

    off = 0
    z = zeros(nz)
    for body in mechanism.bodies
        zi = get_current_state(body)
        ni = state_dimension(body)
        z[off .+ (1:ni)] .= zi; off += ni
    end
    return z
end

function get_next_state(mechanism::Mechanism1170{T,D,NB}) where {T,D,NB}
    nz = sum(state_dimension.(mechanism.bodies))
    z = zeros(nz)
    get_next_state!(z, mechanism)
    return z
end

function get_next_state!(z, mechanism::Mechanism1170{T,D,NB}) where {T,D,NB}
    variables = mechanism.solver.solution.all

    off = 0
    for body in mechanism.bodies
        nz = state_dimension(body)
        get_next_state!(view(z, off .+ (1:nz)), variables, body); off += nz
    end
    return nothing
end

function step!(mechanism::Mechanism1170, z0, u)
    set_current_state!(mechanism, z0)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

function step!(mechanism::Mechanism1170, z0; controller::Function=m->nothing)
    set_current_state!(mechanism, z0)
    controller(mechanism) # sets the control inputs u
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

function simulate!(mechanism::Mechanism1170{T}, z0, H::Int;
        controller::Function=(m,i)->nothing) where T

    storage = Storage(mechanism.dimensions, H, T)
    z = copy(z0)
    for i = 1:H
        z .= step!(mechanism, z, controller=m -> controller(m,i))
        record!(storage, mechanism, i)
    end
    return storage
end






# z1 = rand(12)
# set_current_state!(mech, z1)
# z0 = get_current_state(mech)
# norm(z0 - z1, Inf)






# function get_next_state!(mechanism::Mechanism1170{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nx = 6
#     x = zeros(T, num_bodies*nx)
#     for (i,body) in enumerate(bodies)
#         x[(i-1)*nx .+ (1:nx)] = get_next_state!(mechanism, body)
#     end
#     return x
# end

# function get_next_velocity!(mechanism::Mechanism1170{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nv = 3
#     v = zeros(T, num_bodies*nv)
#     for (i,body) in enumerate(bodies)
#         v[(i-1)*nv .+ (1:nv)] = get_next_velocity!(mechanism, body)
#     end
#     return v
# end

# function get_next_configuration!(mechanism::Mechanism1170{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nq = 3
#     q = zeros(T, num_bodies*nq)
#     for (i,body) in enumerate(bodies)
#         q[(i-1)*nq .+ (1:nq)] = get_next_configuration!(mechanism, body)
#     end
#     return q
# end

# function step!(mechanism::Mechanism1170{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism1170{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism1170{T})
# end
#
# function set_input!(mechanism::Mechanism1170{T})
# end
#
# function set_current_state!(mechanism::Mechanism1170{T})
# end
#
# function set_next_state!(mechanism::Mechanism1170{T})
# end
#
# function get_current_state!(mechanism::Mechanism1170{T})
# end
