mutable struct NodeIndices174
    e::Vector{Int} # equality
    x::Vector{Int} # variables
    θ::Vector{Int} # parameters
end

function NodeIndices174()
    return NodeIndices174(
        collect(1:0),
        collect(1:0),
        collect(1:0),
    )
end

function indexing!(nodes::Vector)
    eoff = 0
    xoff = 0
    θoff = 0
    for node in nodes
        ne = equality_dimension(node)
        nx = variable_dimension(node)
        nθ = parameter_dimension(node)
        node.node_index.e = collect(eoff .+ (1:ne)); eoff += ne
        node.node_index.x = collect(xoff .+ (1:nx)); xoff += nx
        node.node_index.θ = collect(θoff .+ (1:nθ)); θoff += nθ
    end
    return nothing
end
