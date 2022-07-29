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
                @show $var_name
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

macro setuplayer()
    return esc(
        quote
            if symbolic_parsing
                num_vertices = 0
                var_names = Vector{String}()
                graph = SimpleDiGraph()
                svariables = Vector{Vector{Num}}()
                expr = []
            end
        end
    )
end

macro returnlayer()
    return esc(
        quote
            if symbolic_parsing
                return num_vertices, var_names, graph, svariables, expr
            end
        end
    )
end


function f1234(x1::Vector, x11::Vector; symbolic_parsing::Bool=false)
    @setuplayer
    @layer x1
    @layer x11
    x2 = x1 .+ 1
    x21 = x1 .+ 1
    @layer x2
    @layer x21
    x3 = [x2; sum(x11); log(abs(sum(cos.(x2))))]
    @layer x3
    x4 = x3.^2 * x2[end] * x3[end-1] * sum(x2)
    @layer x4
    x5 = [sum(x3) * sum(x4); sin.(x11 * x4[end])]
    @layer x5
    x6 = x5 .+ x21[1] .+ sum(x1)
    x7 = x5 .+ x21[1] .+ sum(x1)
    @layer x6
    @layer x7
    @returnlayer
    return x7
end

n = 2
x1 = rand(n)
x11 = rand(2n)
f1234(x1, x11)
num_vertices0, var_names0, graph0, svariables0, expr0 = f1234(x1, x11; symbolic_parsing=true)
################################################################################
# net
################################################################################
struct SymNet1310{T,NV,NE,NR,NC,NL,Fs,∂Fs}
    x::Vector{Vector{T}} # evaluations
    g::Vector{Vector{T}} # jacobian vector products
    P::Vector{Matrix{T}} # partial jacobians
    J::Vector{Matrix{T}} # jacobians
    graph::SimpleDiGraph
    fcts::Fs
    ∂fcts::∂Fs
    signatures::Vector{Vector{Int}}
end

function generate_symgraph(f::Function, input_dims::Vector{Int}; T=Float64)

    # parse function
    num_vertices, var_names, graph, svariables, expr = f([zeros(T, n) for n in input_dims]...; symbolic_parsing=true)

    # generate expressions
    fcts = []
    ∂fcts = []
    Ps = Vector{Matrix{T}}()
    Js = Vector{Matrix{T}}()
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
    gs = [zeros(T,n) for n in length.(svariables)]

    # number of roots and children
    roots = findall(x -> x==nothing, expr)
    num_roots = length(roots)
    num_children = num_vertices - num_roots

    # function signatures
    signatures = Vector{Vector{Int}}()
    for i = 1:nv(graph)
        ni = indegree(graph, i)
        if ni > 0
            push!(signatures, SVector{1+ni,Int}([i; inneighbors(graph, i)]))
        end
    end

    # leaves of the graph
    leaves = get_leaves(graph)
    num_leaves = length(leaves)

    # Jacobians of all the leaves with respect to all the roots
    for leaf in leaves
        push!(Js, [zeros(T, length(svariables[leaf]), length(svariables[root])) for root in roots]...)
    end

    # types
    NV = num_vertices
    NE = ne(graph)
    NR = num_roots
    NC = num_children
    NL = num_leaves
    types = [T; NV; NE; NR; NC; NL; typeof.([fcts, ∂fcts])]

    return SymNet1310{types...}(xs, gs, Ps, Js, graph, fcts, ∂fcts, signatures)
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
function evaluation20!(symnet::SymNet1310{T,NV,NE,NR}, xroots::Vector) where {T,NV,NE,NR}
    for i in eachindex(xroots)
        symnet.x[i] .= xroots[i]
    end
    for i = NR+1:NV
        evaluation20!(symnet, i)
    end
end

@generated function evaluation20!(symnet::SymNet1310{T,NV,NE,NR}, i::Int) where {T,NV,NE,NR}
    vec = [:(symnet.x[i]); [:(symnet.x[$j]) for j = 1:NV]]
    return :(symnet.fcts[i]($(vec...)))
end

function jvp20!(symnet::SymNet1310{T,NV,NE,NR}, xroots::Vector, v::Vector, leaf::Int=1) where {T,NV,NE,NR}
    evaluation20!(symnet, xroots)
    # traverse the graph upstreams going back to the roots that affect the selected leaf
    for i = 1:NE

    end
end

function jacobian20!(symnet::SymNet1310{T,NV,NE,NR}, xroots::Vector) where {T,NV,NE,NR}
    evaluation20!(symnet, xroots)
    for i = 1:NE
    #     evaluation20!(symnet, i)
    end
end

n = 10
symgraph = generate_symgraph(f1234, [n,n])
signatures0 = symgraph.signatures[1]

xroots0 = [rand(n), rand(n)]

evaluation20!(symgraph, xroots0)
# Main.@code_warntype evaluation20!(symgraph, xroots0)
# @benchmark $evaluation20!($symgraph, $xroots0)
jacobian20!(symgraph, xroots0)
# Main.@code_warntype jacobian20!(symgraph, xroots0)
# @benchmark $jacobian20!($symgraph, $xroots0)
symgraph


norm(symgraph.x[end] - f1234(xroots0...), Inf)
norm(symgraph.x[end], Inf)
norm(f1234(xroots0...), Inf)



symgraph.fcts
symgraph

for i = 1:6
    @show inneighbors(symgraph.graph, i)
end
