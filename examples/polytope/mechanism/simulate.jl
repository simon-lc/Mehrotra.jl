function update_parameters!(mechanism::Mechanism181)
    bodies = mechanism.bodies
    contacts = mechanism.contacts
    solver = mechanism.solver

    off = 0
    for node in [bodies; contacts]
        θ = get_parameters(node)
        nθ = parameter_dimension(node)
        solver.parameters[off .+ (1:nθ)] .= θ; off += nθ
    end
    return nothing
end

function update_nodes!(mechanism::Mechanism181)
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

function set_input!(mechanism::Mechanism181, u)
    off = 0
    nu = length(mechanism.bodies[1].input)
    for body in mechanism.bodies
        body.input .= u[off .+ (1:nu)]; off += nu
    end
    return nothing
end

function get_input(mechanism::Mechanism181{T,D,NB}) where {T,D,NB}
    nu = length(mechanism.bodies[1].input)
    
    off = 0
    u = zeros(nu * NB)
    for body in mechanism.bodies
        u[off .+ (1:nu)] .= body.input; off += nu
    end
    return u
end

function set_current_state!(mechanism::Mechanism181, z)
    off = 0
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    for body in mechanism.bodies
        body.pose .= z[off .+ (1:nx)]; off += nx
        body.velocity .= z[off .+ (1:nv)]; off += nv
    end
    return nothing
end

function get_current_state(mechanism::Mechanism181{T,D,NB}) where {T,D,NB}
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

function get_next_state(mechanism::Mechanism181{T,D,NB}) where {T,D,NB}
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    variables = mechanism.solver.solution.all
    
    off = 0
    z = zeros((nx+nv) * NB)
    for body in mechanism.bodies
        v25 = unpack_variables(variables[body.index.variables], body)
        p2 = body.pose
        timestep = body.timestep
        z[off .+ (1:nx)] .= p2 + timestep[1] .* v25; off += nx
        z[off .+ (1:nv)] .= v25; off += nv
    end
    return z
end


function step!(mechanism::Mechanism181, z0, u)
    set_current_state!(mechanism, z0)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

function step!(mechanism::Mechanism181, z0; controller::Function=m->nothing)
    set_current_state!(mechanism, z0)
    controller(mechanism) # sets the control inputs u
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

mutable struct Storage113{T,H}
    z::Vector{Vector{T}} # H x nz
    u::Vector{Vector{T}} # H x nu
    x::Vector{Vector{Vector{T}}} # H x nb x nx
    v::Vector{Vector{Vector{T}}} # H x nb x nv
    iterations::Vector{Int}
end

function Storage(dim::MechanismDimensions181, H::Int, T=Float64)
    z = [zeros(T, dim.state) for i = 1:H]
    u = [zeros(T, dim.input) for i = 1:H]
    x = [[zeros(T, dim.body_configuration) for j = 1:dim.bodies] for i = 1:H]
    v = [[zeros(T, dim.body_velocity) for j = 1:dim.bodies] for i = 1:H]
    iterations = zeros(Int, H)
    storage = Storage113{T,H}(z, u, x, v, iterations)
    return storage
end

function simulate!(mechanism::Mechanism181{T}, z0, H::Int; 
        controller::Function=(m,i)->nothing) where T

    storage = Storage(mechanism.dimensions, H, T)
    z = copy(z0)
    for i = 1:H
        z .= step!(mechanism, z, controller=m -> controller(m,i))
        record!(storage, mechanism, i)
    end
    return storage
end

function record!(storage::Storage113{T,H}, mechanism::Mechanism181{T,D,NB}, i::Int) where {T,H,D,NB}
    storage.z[i] .= get_current_state(mechanism)
    storage.u[i] .= get_input(mechanism)
    for j = 1:NB
        storage.x[i][j] .= mechanism.bodies[j].pose
        storage.v[i][j] .= mechanism.bodies[j].velocity
    end
    storage.iterations[i] = mechanism.solver.trace.iterations
    return nothing
end



# z1 = rand(12)
# set_current_state!(mech, z1)
# z0 = get_current_state(mech)
# norm(z0 - z1, Inf)






# function get_next_state!(mechanism::Mechanism181{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nx = 6
#     x = zeros(T, num_bodies*nx)
#     for (i,body) in enumerate(bodies)
#         x[(i-1)*nx .+ (1:nx)] = get_next_state!(mechanism, body)
#     end
#     return x
# end

# function get_next_velocity!(mechanism::Mechanism181{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nv = 3
#     v = zeros(T, num_bodies*nv)
#     for (i,body) in enumerate(bodies)
#         v[(i-1)*nv .+ (1:nv)] = get_next_velocity!(mechanism, body)
#     end
#     return v
# end

# function get_next_configuration!(mechanism::Mechanism181{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nq = 3
#     q = zeros(T, num_bodies*nq)
#     for (i,body) in enumerate(bodies)
#         q[(i-1)*nq .+ (1:nq)] = get_next_configuration!(mechanism, body)
#     end
#     return q
# end

# function step!(mechanism::Mechanism181{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism181{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism181{T})
# end
#
# function set_input!(mechanism::Mechanism181{T})
# end
#
# function set_current_state!(mechanism::Mechanism181{T})
# end
#
# function set_next_state!(mechanism::Mechanism181{T})
# end
#
# function get_current_state!(mechanism::Mechanism181{T})
# end
