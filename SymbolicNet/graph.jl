using Symbolics
using BenchmarkTools
using LinearAlgebra
using FiniteDiff
using Test
using Graphs
using GraphRecipes
using Plots
using StaticArrays


################################################################################
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
################################################################################
# TODO need to capture several variables at once in the macro
# TODO need to capture the name of the variables in the macro so that we can
# do differentiation using keyword
# TODO implement hessian vector product, maybe we can reuse the jacobian infrastructure once again
# TODO finish jvp and jacobian

################################################################################
# macro
################################################################################
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
                # add a vertex for the current variable
                num_vertices += 1
                add_vertex!(graph)
                push!(var_names, $var_name)

                # then it means this is not a root of the tree
                # we need to add edges coming from its parents
                if eltype($var) <: Symbolics.Num
                    # add parent edges
                    parent_variables = vcat(Symbolics.get_variables.($var)...)
                    for i = 1:nv(graph)-1
                        # if the current variable depends on variable i we add an edge i -> current
                        if !isempty(intersect(parent_variables, svariables[i]))
                            add_edge!(graph, i, num_vertices)
                            push!(edge_vector, (i, num_vertices))
                            edge_dict[(i, num_vertices)] = length(edge_vector)
                        end
                    end
                    # code generation
                    push!(expr, $var)
                else
                    push!(expr, nothing)
                end

                # display
                if ne(graph) >= 1
                    plt = graphplot(graph, names=1:num_vertices, curvature_scalar=0.01, linewidth=3)
                    display(plt)
                end

                # replace the variable by a simple symbolic vector
                n = length($var)
                svar = Symbolics.variables(Symbol(:x, num_vertices), 1:n)
                push!(svariables, svar)
                $var = svar
            end
        end
    )
end

macro rootlayer()
    return esc(
        quote
            if symbolic_parsing
                num_vertices = 0
                var_names = Vector{String}()
                graph = SimpleDiGraph()
                svariables = Vector{Vector{Num}}()
                expr = []
                edge_vector = Vector{Tuple{Int,Int}}()
                edge_dict = Dict{Tuple{Int,Int},Int}()
            end
        end
    )
end

macro leaflayer()
    return esc(
        quote
            if symbolic_parsing
                return num_vertices, var_names, graph, svariables, expr, edge_vector, edge_dict
            end
        end
    )
end


function f1234(x1::Vector, x11::Vector; symbolic_parsing::Bool=false)
    @rootlayer
    @layer x1
    @layer x11
    x2 = x1 .+ 1
    x21 = x11 .+ 1
    # @layer x2
    # @layer x21
    x3 = [x2; x21[1]; x2[2]]
    @layer x3
    x4 = x3 * x3[end] * x3[end-1] * x3[1]
    @layer x4
    x5 = [x4[1]; sin.(x4 * x4[end])]
    @layer x5
    x6 = x5 .+ x5[1] .+ x5[1]
    x7 = x5 .+ x5[1] .+ x5[1]
    # @layer x6
    @layer x7
    @leaflayer
    return x7
end

# function f1234(x1::Vector, x11::Vector; symbolic_parsing::Bool=false)
#     @rootlayer x1 x11
#     x2 = x1 .+ 1
#     x21 = x1 .+ 1
#     @layer x2 x21
#     x3 = [x2; sum(x11); log(abs(sum(cos.(x2))))]
#     @layer x3
#     x4 = x3.^2 * x2[end] * x3[end-1] * sum(x2)
#     @layer x4
#     x5 = [sum(x3) * sum(x4); sin.(x11 * x4[end])]
#     @layer x5
#     x6 = x5 .+ x21[1] .+ sum(x1)
#     x7 = x5 .+ x21[1] .+ sum(x1)
#     @leaflayer x6 x7
#     return x7
# end

n = 2
x1 = rand(n)
x11 = rand(2n)
f1234(x1, x11)
num_vertices0, var_names0, graph0, svariables0, expr0, expr_edge0, edge_dict0 = f1234(x1, x11; symbolic_parsing=true)

################################################################################
# net
################################################################################
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

function generate_symgraph(f::Function, input_dims::Vector{Int}; T=Float64)
    # parse function
    num_vertices, var_names, graph, svariables, expr, edge_vector, edge_dict = f([zeros(T, n) for n in input_dims]...; symbolic_parsing=true)

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
            parents = svariables[inneighbors(graph, i)]
            push!(fcts,
                # Symbolics.build_function(ex, parents...,
                Symbolics.build_function(ex, svariables...,
                    parallel=Symbolics.SerialForm(),
                    checkbounds=true,
                    expression=Val{false})[2])
            # generate jacobians with respect to each parent
            ∂ex = [Symbolics.jacobian(ex, parent) for parent in parents]
            for ∂exj in ∂ex
                push!(∂fcts,
                    Symbolics.build_function(∂exj, svariables...,
                        parallel=Symbolics.SerialForm(),
                        checkbounds=true,
                        expression=Val{false})[2])
            end
            push!(Ps, [zeros(T, length(svariables[i]), n) for n in length.(parents)]...)
        end
    end

    # storage vectors
    xs = [zeros(T,n) for n in length.(svariables)]
    jvps = [zeros(T,n) for n in length.(svariables)]
    jvp_temps = [zeros(T,n) for n in length.(svariables)]

    # name dictionary to retreive the variables' names easily
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
        push!(Js, [zeros(T, length(svariables[leaf]), length(svariables[root])) for root in roots]...)
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

################################################################################
# evaluation
################################################################################
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


################################################################################
# test
################################################################################
n = 50
symgraph = generate_symgraph(f1234, [n,n])
xroots0 = [rand(n), rand(n)]
xroot10 = xroots0[1]
xroot20 = xroots0[2]

root0 = :x1
leaf0 = :x7

# evaluation
evaluation50!(symgraph, xroots0)
# Main.@code_warntype evaluation50!(symgraph, xroots0)
# @benchmark $evaluation50!($symgraph, $xroots0)
num_allocs = @ballocated $evaluation50!($symgraph, $xroots0)
e1 = get_evaluation(symgraph, leaf0)

e0 = f1234(xroots0...)
norm(e0, Inf)
norm(e1, Inf)
norm(e0 - e1, Inf)


# jacobian vector product
v0 = rand(n+3)
jvp50!(symgraph, v0, xroots0, leaf0)
# Main.@code_warntype jvp50!(symgraph, v0, xroots0, leaf0)
# @benchmark $jvp50!($symgraph, $v0, $xroots0, $leaf0)
num_allocs = @ballocated $jvp50!($symgraph, $v0, $xroots0, $leaf0)

jv1 = symgraph.jvp[1]
jv0 = FiniteDiff.finite_difference_gradient(x1 -> f1234(x1, xroots0[2])'*v0, xroots0[1])
norm(jv0, Inf)
norm(jv1, Inf)
norm(jv0 - jv1, Inf)


# Jacobian
jac0 = zeros(n+3,n)
jacobian50!(jac0, symgraph, xroots0, root0, leaf0)
# Main.@code_warntype jacobian50!(jac0, symgraph, xroots0, root0, leaf0)
# @benchmark $jacobian50!($jac0, $symgraph, $xroots0, $root0, $leaf0)
num_allocs = @ballocated $jacobian50!($jac0, $symgraph, $xroots0, $root0, $leaf0)

jac1 = FiniteDiff.finite_difference_jacobian(x1 -> f1234(x1, xroots0[2]), xroots0[1])
norm(jac0, Inf)
norm(jac1, Inf)
norm(jac0 - jac1, Inf)
