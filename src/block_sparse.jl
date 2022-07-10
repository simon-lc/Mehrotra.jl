struct BlockSparse116{T,N,M}
    matrix::SparseMatrixCSC{T, Int}
    indices::Vector{Vector{Int}}
    name_dict::Dict{Symbol,Int}
    ranges::Vector{Tuple{Vector{Int}, Vector{Int}}}
end

function BlockSparse116(n::Int, m::Int, blocks, ranges;
        names::AbstractVector{Symbol}=[Symbol(:b,i) for i=1:length(blocks)])

    nb = length(blocks)

    # generate and fill the full sparse matrix
    T = eltype(blocks[1])
    matrix = spzeros(T, n, m)
    for i = 1:nb
        matrix[ranges[i]...] .= blocks[i]
    end

    # make sure the is not overlapping blocks
    @assert nnz(matrix) == sum(nnz.(blocks))

    # grad indices for each block
    indices = [zeros(Int, nnz(b)) for b in blocks]
    i_matrix = similar(matrix, Int)
    i_matrix.nzval .= Vector(1:nnz(matrix))
    for i = 1:nb
        indices[i] .= i_matrix[ranges[i]...].nzval
    end

    # name dictionary
    name_dict = Dict{Symbol, Int}()
    for (i,name) in enumerate(names)
        name_dict[name] = i
    end

    return BlockSparse116{T,n,m}(matrix, indices, name_dict, ranges)
end

import Base.fill!
function fill!(block_matrix::BlockSparse116, A1, name)
    matrix = block_matrix.matrix
    indices = block_matrix.indices
    i = block_matrix.name_dict[name]
    @turbo matrix.nzval[indices[i]] = A1.nzval
    return nothing
end
