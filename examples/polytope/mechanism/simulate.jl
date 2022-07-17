function update_parameters!(mechanism::Mechanism182)
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

function update_nodes!(mechanism::Mechanism182)
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

function set_input!(mechanism::Mechanism182, u)
    off = 0
    nu = length(mechanism.bodies[1].input)
    for body in mechanism.bodies
        body.input .= u[off .+ (1:nu)]; off += nu
    end
    return nothing
end

function get_input(mechanism::Mechanism182{T,D,NB}) where {T,D,NB}
    nu = length(mechanism.bodies[1].input)
    
    off = 0
    u = zeros(nu * NB)
    for body in mechanism.bodies
        u[off .+ (1:nu)] .= body.input; off += nu
    end
    return u
end

function set_current_state!(mechanism::Mechanism182, z)
    off = 0
    nx = length(mechanism.bodies[1].pose)
    nv = length(mechanism.bodies[1].velocity)
    for body in mechanism.bodies
        body.pose .= z[off .+ (1:nx)]; off += nx
        body.velocity .= z[off .+ (1:nv)]; off += nv
    end
    return nothing
end

function get_current_state(mechanism::Mechanism182{T,D,NB}) where {T,D,NB}
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

function get_next_state(mechanism::Mechanism182{T,D,NB}) where {T,D,NB}
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

function step!(mechanism::Mechanism182, z0, u)
    set_current_state!(mechanism, z0)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

function step!(mechanism::Mechanism182, z0; controller::Function=m->nothing)
    set_current_state!(mechanism, z0)
    controller(mechanism) # sets the control inputs u
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

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

function Storage(dim::MechanismDimensions182, H::Int, T=Float64)
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

function simulate!(mechanism::Mechanism182{T}, z0, H::Int; 
        controller::Function=(m,i)->nothing) where T

    storage = Storage(mechanism.dimensions, H, T)
    z = copy(z0)
    for i = 1:H
        z .= step!(mechanism, z, controller=m -> controller(m,i))
        record!(storage, mechanism, i)
    end
    return storage
end

function record!(storage::Storage116{T,H}, mechanism::Mechanism182{T,D,NB,NC}, i::Int) where {T,H,D,NB,NC}
    storage.z[i] .= get_current_state(mechanism)
    storage.u[i] .= get_input(mechanism)

    for j = 1:NB
        storage.x[i][j] .= mechanism.bodies[j].pose
        storage.v[i][j] .= mechanism.bodies[j].velocity
    end

    variables = mechanism.solver.solution.all
    parameters = mechanism.solver.parameters
    for (j, contact) in enumerate(mechanism.contacts)
        #############################################
        # contact_point, normal, tangent = contact_frame(contact, pbody, variables, parameters)
        # TODO need to get rid of this into its own function, will be possible when pbody and cbody normal computation will be unified
        pbody = find_body(bodies, contact.parent_name)
        c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = 
            unpack_variables(variables[contact.index.variables], contact)
        friction_coefficient, Ap, bp, Ac, bc = 
            unpack_parameters(parameters[contact.index.parameters], contact)
        vp25 = unpack_variables(variables[pbody.index.variables], pbody)
        pp2, vp15, up2, timestep_p, gravity_p, mass_p, inertia_p = unpack_parameters(parameters[pbody.index.parameters], pbody)
        pp3 = pp2 + timestep_p[1] * vp25        
        normal = -x_2d_rotation(pp3[3:3]) * Ap' * λp
        R = [0 1; -1 0]
        tangent = R * normal
        #####################################################

        storage.contact_point[i][j] .= c
        storage.normal[i][j] .= normal
        storage.tangent[i][j] .= tangent
    end

    storage.variables[i] .= variables
    storage.iterations[i] = mechanism.solver.trace.iterations

    return nothing
end



# z1 = rand(12)
# set_current_state!(mech, z1)
# z0 = get_current_state(mech)
# norm(z0 - z1, Inf)






# function get_next_state!(mechanism::Mechanism182{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nx = 6
#     x = zeros(T, num_bodies*nx)
#     for (i,body) in enumerate(bodies)
#         x[(i-1)*nx .+ (1:nx)] = get_next_state!(mechanism, body)
#     end
#     return x
# end

# function get_next_velocity!(mechanism::Mechanism182{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nv = 3
#     v = zeros(T, num_bodies*nv)
#     for (i,body) in enumerate(bodies)
#         v[(i-1)*nv .+ (1:nv)] = get_next_velocity!(mechanism, body)
#     end
#     return v
# end

# function get_next_configuration!(mechanism::Mechanism182{T}) where T
#     bodies = mechanism.bodies
#     num_bodies = length(bodies)
#     nq = 3
#     q = zeros(T, num_bodies*nq)
#     for (i,body) in enumerate(bodies)
#         q[(i-1)*nq .+ (1:nq)] = get_next_configuration!(mechanism, body)
#     end
#     return q
# end

# function step!(mechanism::Mechanism182{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism182{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism182{T})
# end
#
# function set_input!(mechanism::Mechanism182{T})
# end
#
# function set_current_state!(mechanism::Mechanism182{T})
# end
#
# function set_next_state!(mechanism::Mechanism182{T})
# end
#
# function get_current_state!(mechanism::Mechanism182{T})
# end
