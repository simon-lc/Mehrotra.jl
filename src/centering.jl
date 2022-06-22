function centering(solution::Point228{T}, step::Point228{T}, affine_step_size::T,
        indices::Indices228) where T
    z = solution.duals
    s = solution.slacks

    Δz = step.duals
    Δs = step.slacks

    cone_degree = length(indices.cone_nonnegative) + length(indices.cone_second_order)

    ν = dot(z, s) / cone_degree
    ν_affine = (
        dot(z, s) +
        affine_step_size * dot(Δz, s) +
        affine_step_size * dot(z, Δs) +
        dot(Δz, Δs)
        ) / cone_degree

    σ = clamp(ν_affine / ν, 0.0, 1.0)^3
    central_path_candidate = ν * σ
    return central_path_candidate
end



# solution = solver.solution
# step = solver.data.step
# affine_step_size = 0.1
# indices = solver.indices
# using BenchmarkTools
#
# centering(solution, step, affine_step_size, indices)
# Main.@code_warntype centering(solution, step, affine_step_size, indices)
# @benchmark $centering($solution, $step, $affine_step_size, $indices)
