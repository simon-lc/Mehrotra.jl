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

function dynamics_jacobian_state(dz, mechanism::Mechanism183, z, u, w)
    solver = mechanism.solver
    solver.options.differentiate = true

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:state]
    idx_parameters = mechanism.indices.parameter_state
    idx_solution = mechanism.indices.solution_state
    dx .= solver.data.solution_sensitivity[idx_solution, idx_parameters]
    return nothing
end

function dynamics_jacobian_input(du, mechanism::Mechanism183, z, u, w)
    solver = mechanism.solver
    solver.options.differentiate = true

    set_current_state!(mechanism, z)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(solver)

    # idx_parameters = solver.indices.parameter_keywords[:input]
    idx_parameters = mechanism.indices.input
    idx_solution = mechanism.indices.solution_state
    dx .= solver.data.solution_sensitivity[idx_solution, idx_parameters]
    return nothing
end
