# solver
function initialize!(solver::Solver228, guess)
    # variables
    solver.solution.all .= guess
    return
end

function initialize_primals!(solver)
    solver.solution.primals .= 0.0
    return
end

function initialize_duals!(solver)
    initialize_cone!(
        solver.solution.duals,
        solver.indices.cone_nonnegative,
        solver.indices.cone_second_order)
    return
end

function initialize_slacks!(solver)
    initialize_cone!(
        solver.solution.slacks,
        solver.indices.cone_nonnegative,
        solver.indices.cone_second_order)
    return
end

function initialize_interior_point!(solver)
    solver.central_path[1] = solver.options.central_path_initial
    solver.fraction_to_boundary[1] = max(0.99, 1.0 - solver.central_path[1])
    return
end
