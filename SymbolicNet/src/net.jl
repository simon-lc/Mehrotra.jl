struct SymNet1440{T,NV,NE,NR,NC,NL,Fs,∂Fs}
    x::Vector{Vector{T}} # evaluations
    jvp::Vector{Vector{T}} # jacobian vector products
    jvp_temp::Vector{Vector{T}} # jacobian vector products temporary variables
    P::Vector{Matrix{T}} # partial jacobians
    J::Vector{Matrix{T}} # jacobians
    graph::SimpleDiGraph
    fcts::Fs
    ∂fcts::∂Fs
    name_dict::Dict{Symbol,Int}
    leaves::Vector{Int}
    edge_vector::Vector{Tuple{Int,Int}}
    edge_dict::Dict{Tuple{Int,Int},Int}
    vertex_streams::Vector{Vector{Int}} # a sequence per leaf, indicating a BFS order to execute the jacobian vector product
    edge_streams::Vector{Vector{Int}} # a sequence per leaf, indicating a BFS order to execute the jacobian vector product
end

function generate_symgraph(f::Function, root_dims::Vector{Int}; T=Float64)
    # parse function
    var_names, graph, variables, expr, edge_vector, edge_dict = f([zeros(T, n) for n in root_dims]...; symbolic_parsing=true)

    # generate expressions
    fcts = []
    ∂fcts = []
    Js = Vector{Matrix{T}}()
    Ps = Vector{Matrix{T}}()
    for (i,ex) in enumerate(expr)
        if ex == nothing
            push!(fcts, nothing)
        else
            # generate function where parents are inputs
            parents = variables[inneighbors(graph, i)]
            push!(fcts,
                # Symbolics.build_function(ex, parents...,
                Symbolics.build_function(ex, variables...,
                    parallel=Symbolics.SerialForm(),
                    checkbounds=true,
                    expression=Val{false})[2])
            # generate jacobians with respect to each parent
            ∂ex = [Symbolics.jacobian(ex, parent) for parent in parents]
            for ∂exj in ∂ex
                push!(∂fcts,
                    Symbolics.build_function(∂exj, variables...,
                        parallel=Symbolics.SerialForm(),
                        checkbounds=true,
                        expression=Val{false})[2])
            end
            push!(Ps, [zeros(T, length(variables[i]), n) for n in length.(parents)]...)
        end
    end

    # storage vectors
    xs = [zeros(T,n) for n in length.(variables)]
    jvps = [zeros(T,n) for n in length.(variables)]
    jvp_temps = [zeros(T,n) for n in length.(variables)]

    # name dictionary to retreive the variables' names easily
    num_vertices = nv(graph)
    name_dict = Dict{Symbol,Int}(Symbol(var_names[i]) => i for i = 1:num_vertices)

    # number of roots and children
    roots = findall(x -> x==nothing, expr)
    num_roots = length(roots)
    num_children = num_vertices - num_roots

    # leaves of the graph
    leaves = get_leaves(graph)
    num_leaves = length(leaves)

    # Jacobians of all the leaves with respect to all the roots
    for leaf in leaves
        push!(Js, [zeros(T, length(variables[leaf]), length(variables[root])) for root in roots]...)
    end

    # jacobian vector product streams
    vertex_streams, edge_streams = jacobian_stream(graph, leaves, edge_dict)

    # types
    NV = num_vertices
    NE = ne(graph)
    NR = num_roots
    NC = num_children
    NL = num_leaves
    types = [T; NV; NE; NR; NC; NL; typeof.([fcts, ∂fcts])]

    return SymNet1440{types...}(xs, jvps, jvp_temps, Ps, Js, graph, fcts, ∂fcts,
        name_dict, leaves, edge_vector, edge_dict, vertex_streams, edge_streams)
end

function jacobian_stream(graph::SimpleDiGraph, leaves::Vector{Int}, edge_dict::Dict)
    vertex_streams = Vector{Vector{Int}}()
    edge_streams = Vector{Vector{Int}}()
    for leaf in leaves
        vertex_stream, edge_stream = jacobian_stream(graph, leaf, edge_dict)
        push!(vertex_streams, vertex_stream)
        push!(edge_streams, edge_stream)
    end
    return vertex_streams, edge_streams
end

function jacobian_stream(graph::SimpleDiGraph, leaf::Int, edge_dict::Dict)
    vertex_stream = [leaf]
    edge_stream = Vector{Int}()
    vchild = leaf

    vstack = copy(inneighbors(graph, leaf))
    estack = [(p, leaf) for p in inneighbors(graph, leaf)]
    while !isempty(vstack)
        v = popfirst!(vstack)
        e = popfirst!(estack)
        push!(vertex_stream, v)
        push!(edge_stream, edge_dict[e])

        parents = inneighbors(graph, v)
        push!(vstack, parents...)
        push!(estack, [(p, v) for p in parents]...)
        vchild = v
    end
    return vertex_stream, edge_stream
end

function get_leaves(g::SimpleDiGraph)
    l = Vector{Int}()
    num_vertices = nv(g)
    for i = 1:num_vertices
        isempty(outneighbors(g, i)) && push!(l, i)
    end
    return l
end
