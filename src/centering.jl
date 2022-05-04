function centering(solution::Point228{T}, step::Point228{T}, affine_step_size::T, indices::Indices228) where T
    z = solution.duals
    s = solution.slacks

    Δz = step.duals
    Δs = step.slacks

    cone_degree = length(indices.cone_nonnegative) + length(indices.cone_second_order)

    ν = dot(z, s) / cone_degree
    ν_affine = dot(z + affine_step_size * Δz, s + affine_step_size * Δs) / cone_degree

    σ = clamp(ν_affine / ν, 0.0, 1.0)^3
    central_path_candidate = ν * σ
    return central_path_candidate
end
