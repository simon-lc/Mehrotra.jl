struct CentralPath{T}
    central_path::Vector{T}
    target_central_path::Vector{T}
    tolerance_central_path::Vector{T}
    zero_central_path::Vector{T}
    neutral_central_path::Vector{T}
end

function CentralPath(idx_nn::Vector{Int}, idx_soc::Vector{Vector{Int}}, complementarity_tolerance, T=Float64)

    num_cone = length(idx_nn) + sum(length.(idx_soc))
    neutral = ones(T, num_cone)
    for soc in idx_soc
        neutral[soc[2:end]] .= 0
    end

    CentralPath{T}(
        0.1*neutral,
        0.1*neutral,
        complementarity_tolerance .* neutral,
        zeros(T,num_cone),
        neutral
    )
end
