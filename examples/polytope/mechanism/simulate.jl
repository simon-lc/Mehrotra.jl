function get_next_state!(mechanism::Mechanism175{T}) where T
    bodies = mechanism.bodies
    num_bodies = length(bodies)
    nx = 6
    x = zeros(T, num_bodies*nx)
    for (i,body) in enumerate(bodies)
        x[(i-1)*nx .+ (1:nx)] = get_next_state!(mechanism, body)
    end
    return x
end

function get_next_velocity!(mechanism::Mechanism175{T}) where T
    bodies = mechanism.bodies
    num_bodies = length(bodies)
    nv = 3
    v = zeros(T, num_bodies*nv)
    for (i,body) in enumerate(bodies)
        v[(i-1)*nv .+ (1:nv)] = get_next_velocity!(mechanism, body)
    end
    return v
end

function get_next_configuration!(mechanism::Mechanism175{T}) where T
    bodies = mechanism.bodies
    num_bodies = length(bodies)
    nq = 3
    q = zeros(T, num_bodies*nq)
    for (i,body) in enumerate(bodies)
        q[(i-1)*nq .+ (1:nq)] = get_next_configuration!(mechanism, body)
    end
    return q
end



# function step!(mechanism::Mechanism175{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism175{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism175{T})
# end
#
# function set_input!(mechanism::Mechanism175{T})
# end
#
# function set_current_state!(mechanism::Mechanism175{T})
# end
#
# function set_next_state!(mechanism::Mechanism175{T})
# end
#
# function get_current_state!(mechanism::Mechanism175{T})
# end
