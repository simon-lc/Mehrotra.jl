function update_parameters!(mechanism::Mechanism183)
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

function update_nodes!(mechanism::Mechanism183)
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

function set_input!(mechanism::Mechanism183, u)
    off = 0
    nu = length(mechanism.bodies[1].input)
    for body in mechanism.bodies
        body.input .= u[off .+ (1:nu)]; off += nu
    end
    return nothing
end

function get_input(mechanism::Mechanism183{T,D,NB}) where {T,D,NB}
    nu = length(mechanism.bodies[1].input)

    off = 0
    u = zeros(nu * NB)
    for body in mechanism.bodies
        u[off .+ (1:nu)] .= body.input; off += nu
    end
    return u
end

function set_current_state!(mechanism::Mechanism183, z)
    off = 0
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    for body in mechanism.bodies
        body.pose .= z[off .+ (1:nx)]; off += nx
        body.velocity .= z[off .+ (1:nv)]; off += nv
    end
    return nothing
end

function get_current_state(mechanism::Mechanism183{T,D,NB}) where {T,D,NB}
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)

    off = 0
    z = zeros((nx+nv) * NB)
    for body in mechanism.bodies
        z[off .+ (1:nx)] .= body.pose; off += nx
        z[off .+ (1:nv)] .= body.velocity; off += nv
    end
    return z
end

function get_next_state(mechanism::Mechanism183{T,D,NB}) where {T,D,NB}
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    z = zeros((nx+nv) * NB)
    get_next_state!(z, mechanism)
    return z
end

function get_next_state!(z, mechanism::Mechanism183{T,D,NB}) where {T,D,NB}
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    variables = mechanism.solver.solution.all

    off = 0
    for body in mechanism.bodies
        v25 = unpack_variables(variables[body.index.variables], body)
        p2 = body.pose
        timestep = body.timestep
        z[off .+ (1:nx)] .= p2 + timestep[1] .* v25; off += nx
        z[off .+ (1:nv)] .= v25; off += nv
    end
    return nothing
end

function step!(mechanism::Mechanism183, z0, u)
    set_current_state!(mechanism, z0)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

function step!(mechanism::Mechanism183, z0; controller::Function=m->nothing)
    set_current_state!(mechanism, z0)
    controller(mechanism) # sets the control inputs u
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

function simulate!(mechanism::Mechanism183{T}, z0, H::Int;
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






# function get_next_state!(mechanism::Mechanism183{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nx = 6
#     x = zeros(T, num_bodies*nx)
#     for (i,body) in enumerate(bodies)
#         x[(i-1)*nx .+ (1:nx)] = get_next_state!(mechanism, body)
#     end
#     return x
# end

# function get_next_velocity!(mechanism::Mechanism183{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nv = 3
#     v = zeros(T, num_bodies*nv)
#     for (i,body) in enumerate(bodies)
#         v[(i-1)*nv .+ (1:nv)] = get_next_velocity!(mechanism, body)
#     end
#     return v
# end

# function get_next_configuration!(mechanism::Mechanism183{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nq = 3
#     q = zeros(T, num_bodies*nq)
#     for (i,body) in enumerate(bodies)
#         q[(i-1)*nq .+ (1:nq)] = get_next_configuration!(mechanism, body)
#     end
#     return q
# end

# function step!(mechanism::Mechanism183{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism183{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism183{T})
# end
#
# function set_input!(mechanism::Mechanism183{T})
# end
#
# function set_current_state!(mechanism::Mechanism183{T})
# end
#
# function set_next_state!(mechanism::Mechanism183{T})
# end
#
# function get_current_state!(mechanism::Mechanism183{T})
# end
