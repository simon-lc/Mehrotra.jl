function centering!(central_path::Vector{T}, solution::Point{T}, step::Point{T},
        affine_step_size::Vector{T}, indices::Indices; options::Options=Options()) where T
    z = solution.duals
    s = solution.slacks

    Δz = step.duals
    Δs = step.slacks

    if options.complementarity_decoupling
        # non negative cone
        for (i,ii) in enumerate(indices.cone_nonnegative)
            cone_degree = 1
            ν = dot(z[ii], s[ii]) / cone_degree
            ν_affine = (
                dot(z[ii], s[ii]) +
                affine_step_size[ii] * dot(Δz[ii], s[ii]) +
                affine_step_size[ii] * dot(z[ii], Δs[ii]) +
                affine_step_size[ii]^2 * dot(Δz[ii], Δs[ii])
                ) / cone_degree
            σ = clamp(ν_affine / ν, 0.0, 1.0)^3
            candidate_central_path = ν * σ
            central_path[ii] = max(candidate_central_path, options.complementarity_tolerance)
        end

        # non negative cone
        for ind in indices.cone_second_order
            (length(ind) == 0) && continue
            cone_degree = 1
            ν = 0.0
            ν_affine = 0.0
            for i in ind
                ν += (z[i] * s[i]) / cone_degree
                ν_affine += (
                    z[i] * s[i] +
                    affine_step_size[i] * Δz[i] * s[i] +
                    affine_step_size[i] * z[i] * Δs[i] +
                    affine_step_size[i]^2 * Δz[i] * Δs[i]
                    ) / cone_degree
            end

            σ = clamp(ν_affine / ν, 0.0, 1.0)^3
            candidate_central_path = ν * σ
            central_path[ind[1]] = max(candidate_central_path, options.complementarity_tolerance)
        end

    else
        cone_degree = length(indices.cone_nonnegative) + length(indices.cone_second_order)
        num_cone = length(indices.duals)

        ν = dot(z, s) / cone_degree
        ν_affine = 0.0
        for i = 1:num_cone
            ν_affine += z[i] * s[i]
            ν_affine += affine_step_size[i] * Δz[i] * s[i]
            ν_affine += affine_step_size[i] * z[i] * Δs[i]
            ν_affine += affine_step_size[i]^2 * Δz[i] * Δs[i]
        end
        ν_affine /= cone_degree

        σ = clamp(ν_affine / ν, 0.0, 1.0)^3
        candidate_central_path = ν * σ
        # preserve the neutral vector structure e.g. [1, 0, 0] for a second order cone
        central_path .= max(candidate_central_path, options.complementarity_tolerance) * (central_path .> 0)
    end

    return nothing
end

a = [1e-3, 0]
(a .> 0) .* 10.0
# solver.indices
#
# central_path = solver.central_paths.target_central_path
# solution = solver.solution
# step = solver.data.step
# affine_step_size = solver.step_sizes.affine_step_size
# indices = solver.indices
# options = solver.options
#
# using BenchmarkTools
#
# centering!(central_path, solution, step, affine_step_size, indices, options=options)
# Main.@code_warntype centering!(central_path, solution, step, affine_step_size, indices, options=options)
# @benchmark $centering!($central_path, $solution, $step, $affine_step_size, $indices, options=$options)
