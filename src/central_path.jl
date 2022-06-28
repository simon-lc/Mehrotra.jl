struct CentralPath228{T}
    central_path::Vector{T}
    target_central_path::Vector{T}
    tolerance_central_path::Vector{T}
    zero_central_path::Vector{T}
end

function CentralPath228(num_cone::Int, complementarity_tolerance, T=Float64)

    CentralPath228{T}(
        0.1*ones(T,num_cone),
        0.1*ones(T,num_cone),
        complementarity_tolerance .* ones(T,num_cone),
        zeros(T,num_cone),
    )
end
