function cone_violation(solver::Solver228)
    cone_target = solver.problem.cone_target
    cone_product = solver.data.residual.cone_product
    complementarity_tolerance = solver.options.complementarity_tolerance

    # the one part: avoid penalizing over-satisfaction of the cone constraints
    violation_1 = norm(cone_target .* max.(0.0, cone_product), Inf)
    # the zero part
    violation_0 = norm((1.0 .- cone_target) .* cone_product, Inf)
    return max(violation_0, violation_1)
end
