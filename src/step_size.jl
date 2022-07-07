struct StepSize{T}
    step_size::Vector{T}
    affine_step_size::Vector{T}
end

function StepSize(num_cone, T=Float64)
    StepSize{T}(
        ones(T,num_cone),
        ones(T,num_cone),
    )
end
