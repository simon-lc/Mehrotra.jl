struct Point{T}
    all::Vector{T}
    primals::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    duals::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    slacks::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    equality::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    cone_product::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
end

function Point(dims::Dimensions, idx::Indices)
    x = zeros(dims.variables)
    y = @views x[idx.primals]
    z = @views x[idx.duals]
    s = @views x[idx.slacks]
    equality = @views x[idx.equality]
    cone_product = @views x[idx.cone_product]
    return Point(x, y, z, s, equality, cone_product)
end
