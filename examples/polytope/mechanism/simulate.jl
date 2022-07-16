function update_parameters!(mechanism::Mechanism177)
    bodies = mechanism.bodies
    contacts = mechanism.contacts
    solver = mechanism.solver

    off = 0
    for node in [bodies; contacts]
        θ = get_parameters(node)
        nθ = parameter_dimension(node)
        solver.parameters[off .+ (1:nθ)] .= θ
    end
    return nothing
end

function set_input!(mechanism::Mechanism177, u)
    off = 0
    nu = length(mechanism.bodies[1].input)
    for body in mechanism.bodies
        body.input .= u[off .+ (1:nu)]; off += nu
    end
    return nothing
end

function set_current_state!(mechanism::Mechanism177, z)
    off = 0
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    for body in mechanism.bodies
        body.pose .= z[off .+ (1:nx)]; off += nx
        body.velocity .= z[off .+ (1:nv)]; off += nv
    end
    return nothing
end

function get_current_state(mechanism::Mechanism177{T,D,NB}) where {T,D,NB}
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

function get_next_state(mechanism::Mechanism177{T,D,NB}) where {T,D,NB}
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    variables = mechanism.solver.solution.all
    
    off = 0
    z = zeros((nx+nv) * NB)
    for body in mechanism.bodies
        v25 = unpack_variables(variables, body.index.variables)
        p2 = body.pose
        timestep = body.timestep
        @warn "finish"
        z[off .+ (1:nx)] .= body.pose; off += nx
        z[off .+ (1:nv)] .= body.velocity; off += nv
    end
    return z
end


# z1 = rand(12)
# set_current_state!(mech, z1)
# z0 = get_current_state(mech)
# norm(z0 - z1, Inf)






# function get_next_state!(mechanism::Mechanism177{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nx = 6
#     x = zeros(T, num_bodies*nx)
#     for (i,body) in enumerate(bodies)
#         x[(i-1)*nx .+ (1:nx)] = get_next_state!(mechanism, body)
#     end
#     return x
# end

# function get_next_velocity!(mechanism::Mechanism177{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nv = 3
#     v = zeros(T, num_bodies*nv)
#     for (i,body) in enumerate(bodies)
#         v[(i-1)*nv .+ (1:nv)] = get_next_velocity!(mechanism, body)
#     end
#     return v
# end

# function get_next_configuration!(mechanism::Mechanism177{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nq = 3
#     q = zeros(T, num_bodies*nq)
#     for (i,body) in enumerate(bodies)
#         q[(i-1)*nq .+ (1:nq)] = get_next_configuration!(mechanism, body)
#     end
#     return q
# end

# function step!(mechanism::Mechanism177{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism177{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism177{T})
# end
#
# function set_input!(mechanism::Mechanism177{T})
# end
#
# function set_current_state!(mechanism::Mechanism177{T})
# end
#
# function set_next_state!(mechanism::Mechanism177{T})
# end
#
# function get_current_state!(mechanism::Mechanism177{T})
# end
