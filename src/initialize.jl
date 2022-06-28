# solver
function initialize!(solver::Solver228, variables)
    # variables
    solver.solution.all .= variables
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
    solver.central_paths.central_path .= solver.options.central_path_initial
    solver.fraction_to_boundary .= max.(0.99, 1.0 .- solver.central_paths.central_path)
    return
end

# solver
# variables = ones(solver.dimensions.variables)
#
# initialize!(solver, variables)
# Main.@code_warntype initialize!(solver, variables)
# @benchmark $initialize!($solver, $variables)
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
