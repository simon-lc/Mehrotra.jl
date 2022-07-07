struct StepSize228{T}
    step_size::Vector{T}
    affine_step_size::Vector{T}
end

function StepSize228(num_cone, T=Float64)
    StepSize228{T}(
        ones(T,num_cone),
        ones(T,num_cone),
    )
end
