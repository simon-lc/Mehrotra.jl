mutable struct NodeIndices171
    e::Vector{Int}
    x::Vector{Int}
    θ::Vector{Int}
end

function NodeIndices171()
    return NodeIndices171(
        collect(1:0),
        collect(1:0),
        collect(1:0),
    )
end
