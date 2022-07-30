function evaluation50!(symnet::SymNet1440{T,NV,NE,NR}, xroots::Vector) where {T,NV,NE,NR}
    for i in eachindex(xroots)
        symnet.x[i] .= xroots[i]
    end
    for i = NR+1:NV
        evaluation50!(symnet, i)
    end
end

function get_evaluation(symnet::SymNet1440, leaf::Symbol)
    ileaf = symnet.name_dict[leaf]
    return symnet.x[ileaf]
end

@generated function evaluation50!(symnet::SymNet1440{T,NV,NE,NR}, i::Int) where {T,NV,NE,NR}
    vec = [:(symnet.x[i]); [:(symnet.x[$j]) for j = 1:NV]]
    return :(symnet.fcts[i]($(vec...)))
end

function partial_jacobian50!(symnet::SymNet1440{T,NV,NE}) where {T,NV,NE}
    for i = 1:NE
        partial_jacobian50!(symnet, i)
    end
    return nothing
end

@generated function partial_jacobian50!(symnet::SymNet1440{T,NV,NE,NR}, i::Int) where {T,NV,NE,NR}
    vec = [:(symnet.x[$j]) for j = 1:NV]
    return :(symnet.∂fcts[i](symnet.P[i], $(vec...)))
end

function jvp50!(symnet::SymNet1440{T,NV,NE,NR}, vec::Vector, xroots::Vector, leaf::Symbol;
        evaluation::Bool=true,
        partial_jacobian::Bool=true,
        ) where {T,NV,NE,NR}
    # forward pass
    evaluation && evaluation50!(symnet, xroots)

    # compute the partial jacobians
    partial_jacobian && partial_jacobian50!(symnet)

    # reset the jacobian vector products and intialize the leaf
    ileaf = symnet.name_dict[leaf]
    symnet.jvp[ileaf] .= vec
    for i = 1:NV
        (i != ileaf) && fill!(symnet.jvp[i], 0.0)
    end

    # traverse the graph upstreams going back to the roots that affect the selected leaf
    # contributions from all the routes guiding to the same root add up
    jleaf = findfirst(x -> x == ileaf, symnet.leaves)
    vertex_stream = symnet.vertex_streams[jleaf]
    edge_stream = symnet.edge_streams[jleaf]
    for i = 1:length(vertex_stream)-1
        ei = edge_stream[i]
        src = vertex_stream[i+1]
        dst = symnet.edge_vector[ei][2]
        mul!(symnet.jvp_temp[src], transpose(symnet.P[ei]), symnet.jvp[dst])
        symnet.jvp[src] .+= symnet.jvp_temp[src]
    end
end

function jacobian50!(jac::Matrix, symnet::SymNet1440{T,NV,NE,NR}, xroots::Vector,
        root::Symbol,
        leaf::Symbol;
        evaluation::Bool=true,
        partial_jacobian::Bool=true,
        ) where {T,NV,NE,NR}

    # forward pass
    evaluation && evaluation50!(symnet, xroots)

    # compute the partial jacobians
    partial_jacobian && partial_jacobian50!(symnet)

    ileaf = symnet.name_dict[leaf]
    iroot = symnet.name_dict[root]
    nleaf = length(symnet.jvp[ileaf])
    for i = 1:nleaf
        vec = symnet.jvp[ileaf]
        fill!(vec, 0.0)
        vec[i] = 1.0
        jvp50!(symnet, vec, xroots, leaf, evaluation=false, partial_jacobian=false)
        jac[i,:] .= symnet.jvp[iroot]
    end

    return nothing
end
