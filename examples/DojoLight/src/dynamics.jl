function dynamics(z1, mechanism::Mechanism183, z, u, w)
    solver = mechanism.solver
    solver.options.differentiate = false

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    get_next_state!(z1, mechanism)
    return nothing
end

function dynamics_jacobian_state(dz, mechanism::Mechanism183{T,D,NB}, z, u, w) where {T,D,NB}
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:state]
    idx_parameters = mechanism.indices.parameter_state
    idx_solution = mechanism.indices.solution_state
    idx_velocity = vcat([6(i-1) .+ (4:6) for i=1:NB]...)
    idx_pose = vcat([6(i-1) .+ (1:3) for i=1:NB]...)
    dz[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution, idx_parameters]
    dz[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution, idx_parameters]
    return nothing
end

function dynamics_jacobian_input(du, mechanism::Mechanism183{T,D,NB}, z, u, w) where {T,D,NB}
    solver = mechanism.solver
    solver.options.differentiate = true
    timestep = mechanism.bodies[1].timestep[1]

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:input]
    idx_parameters = mechanism.indices.input
    idx_solution = mechanism.indices.solution_state
    idx_velocity = vcat([6(i-1) .+ (4:6) for i=1:NB]...)
    idx_pose = vcat([6(i-1) .+ (1:3) for i=1:NB]...)
    du[idx_velocity,:] .= solver.data.solution_sensitivity[idx_solution, idx_parameters]
    du[idx_pose,:] .= timestep * solver.data.solution_sensitivity[idx_solution, idx_parameters]
    return nothing
end
