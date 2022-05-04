struct Point228{T}
    all::Vector{T}
    primals::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    duals::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    slacks::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    equality::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    cone_product::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
end

function Point(dims::Dimensions228, idx::Indices228)
    x = zeros(dims.variables)
    y = @views x[idx.primals]
    z = @views x[idx.duals]
    s = @views x[idx.slacks]
    equality = @views x[idx.equality]
    cone_product = @views x[idx.cone_product]
    return Point228(x, y, z, s, equality, cone_product)
end
