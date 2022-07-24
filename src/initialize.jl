function set_variables!(solver::Solver, variables)
    # variables
    solver.solution.all .= variables
    # solution should be recomputed
    solver.consistency.solved .= false
    return
end

function set_parameters!(solver::Solver, parameters)
    # variables
    solver.parameters .= parameters
    # solution and gradients should be recomputed
    solver.consistency.solved .= false
    set_bool!(solver.consistency.differentiated, false)
end

function initialize_primals!(solver)
    solver.solution.primals .= 0.0
    # solution should be recomputed
    solver.consistency.solved .= false
    return
end

function initialize_duals!(solver)
    initialize_cone!(
        solver.solution.duals,
        solver.indices.cone_nonnegative,
        solver.indices.cone_second_order)
    # solution should be recomputed
    solver.consistency.solved .= false
    return
end

function initialize_slacks!(solver)
    initialize_cone!(
        solver.solution.slacks,
        solver.indices.cone_nonnegative,
        solver.indices.cone_second_order)
    # solution should be recomputed
    solver.consistency.solved .= false
    return
end

function initialize_interior_point!(solver)
    options = solver.options
    central_paths = solver.central_paths
    central_paths.central_path .= options.central_path_initial .* central_paths.neutral_central_path
    central_paths.target_central_path .= options.central_path_initial .* central_paths.neutral_central_path
    central_paths.tolerance_central_path .= options.complementarity_tolerance .* central_paths.neutral_central_path
    central_paths.correction_central_path .= options.complementarity_tolerance .* central_paths.neutral_central_path
    # solver.fraction_to_boundary .= max.(0.99, 1.0 .- central_paths.central_path)
    return
end

function initialize_solver!(solver)
    initialize_primals!(solver)
    initialize_duals!(solver)
    initialize_slacks!(solver)
    initialize_interior_point!(solver)
    return nothing
end


# solver
# variables = ones(solver.dimensions.variables)
#
# set_variables!(solver, variables)
# Main.@code_warntype set_variables!(solver, variables)
# @benchmark $set_variables!($solver, $variables)
#
# initialize_primals!(solver)
# Main.@code_warntype initialize_primals!(solver)
# @benchmark $initialize_primals!($solver)
#
# initialize_duals!(solver)
# Main.@code_warntype initialize_duals!(solver)
# @benchmark $initialize_duals!($solver)
#
# initialize_slacks!(solver)
# Main.@code_warntype initialize_slacks!(solver)
# @benchmark $initialize_slacks!($solver)
#
# initialize_interior_point!(solver)
# Main.@code_warntype initialize_interior_point!(solver)
# @benchmark $initialize_interior_point!($solver)
