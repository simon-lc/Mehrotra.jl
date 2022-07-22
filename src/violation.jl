function violation(problem::ProblemData, κ::Vector)
    equality_violation = norm(problem.equality_constraint, Inf)
    cone_product_violation = 0.0
    for (i,κi) in enumerate(κ)
        cone_product_violation = max(cone_product_violation, problem.cone_product[i] - κi)
    end
    return equality_violation, cone_product_violation
end

function violation(solver::Solver)
    violation(solver.problem, solver.central_paths.tolerance_central_path)
end

# violation(solver)
# Main.@code_warntype violation(solver)
# @benchmark $violation($solver)
