function add_layer!(var, var_name, var_names::Vector{String}, graph::SimpleDiGraph,
        variables::Vector, expr, edge_vector::Vector, edge_dict::Dict)

    # add a vertex for the current variable
    add_vertex!(graph)
    num_vertices = nv(graph)
    push!(var_names, var_name)

    # then it means this is not a root of the tree
    # we need to add edges coming from its parents
    if eltype(var) <: Symbolics.Num
        # add parent edges
        parent_variables = vcat(Symbolics.get_variables.(var)...)
        for i = 1:nv(graph)-1
            # if the current variable depends on variable i we add an edge i -> current
            if !isempty(intersect(parent_variables, variables[i]))
                add_edge!(graph, i, num_vertices)
                push!(edge_vector, (i, num_vertices))
                edge_dict[(i, num_vertices)] = length(edge_vector)
            end
        end
        # code generation
        push!(expr, var)
    else
        push!(expr, nothing)
    end

    # # display
    # if ne(graph) >= 1
    #     plt = graphplot(graph, names=1:num_vertices, curvature_scalar=0.01, linewidth=3)
    #     display(plt)
    # end

    # replace the variable by a simple symbolic vector
    n = length(var)
    svar = Symbolics.variables(Symbol(:x, num_vertices), 1:n)
    push!(variables, svar)
    return svar
end

"""
    This macro picks up a variable xk midway through the function execution store
    symbolic expression. Then its (complex) expression xk = fct(x0, xk-1)
    is replaced by a simple redefinition xk = [xk0, xk1, xk2].

    This need to be placed after the definition of xk and ideally before xk is
    used in any other expression.
"""
macro layer(var)
    var_name = string(var)
    return esc(
        quote
            if symbolic_parsing
                $var = add_layer!($var, $var_name,
                    var_names,
                    graph,
                    variables,
                    expr,
                    edge_vector,
                    edge_dict)
            end
        end
    )
end

macro rootlayer(var)
    var_name = string(var)
    return esc(
        quote
            if symbolic_parsing
                var_names = Vector{String}()
                graph = SimpleDiGraph()
                variables = Vector{Vector{Num}}()
                expr = []
                edge_vector = Vector{Tuple{Int,Int}}()
                edge_dict = Dict{Tuple{Int,Int},Int}()

                $var = add_layer!($var, $var_name,
                    var_names,
                    graph,
                    variables,
                    expr,
                    edge_vector,
                    edge_dict)
            end
        end
    )
end

macro leaflayer(var)
    var_name = string(var)
    return esc(
        quote
            if symbolic_parsing
                $var = add_layer!($var, $var_name,
                    var_names,
                    graph,
                    variables,
                    expr,
                    edge_vector,
                    edge_dict)
                return var_names, graph, variables, expr, edge_vector, edge_dict
            end
        end
    )
end
