function cone_violation(solver::Solver)
    neutral_central_path = solver.central_paths.neutral_central_path
    cone_product = solver.data.residual.cone_product
    complementarity_tolerance = solver.options.complementarity_tolerance

    violation = 0.0
    for i = 1:solver.dimensions.cone
        # the one part: avoid penalizing over-satisfaction of the cone constraints
        if neutral_central_path[i] == 1
            # violation = max(violation, abs(max(0, cone_product[i])))
            violation = max(violation, abs(cone_product[i]))
        # the zero part
        else
            violation = max(violation, abs(cone_product[i]))
        end
    end
    return violation
end

function violation(problem::ProblemData, κ::Vector)
    equality_violation = norm(problem.equality_constraint, Inf)
    cone_product_violation = norm(problem.cone_product .- κ, Inf)
    return equality_violation, cone_product_violation
end

# cone_violation(solver)
# Main.@code_warntype cone_violation(solver)
# @benchmark $cone_violation($solver)
