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
# macro
################################################################################
"""
    This macro splits the function into layers and pass variables as input to a layer
    and compute the output of the layer. This generate simple exparessions.
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
    # @layer x2
    # @layer x21
    x3 = [x2; sum(x11); log(abs(sum(cos.(x2))))]
    @layer x3
    x4 = x3.^2 * x2[end] * x3[end-1] * sum(x2)
    @layer x4
    x5 = [sum(x3) * sum(x4); sin.(x11 * x4[end])]
    # @layer x5
    x6 = x5 .+ x21[1] .+ sum(x1)
    @layer x6
    @returnlayer
    return x6
end

n = 2
x1 = rand(n)
x11 = rand(2n)
f1234(x1, x11)
num_vertices0, var_names0, graph0, svariables0, expr0 = f1234(x1, x11; symbolic_parsing=true)
################################################################################
# net
################################################################################
struct SymNet1290{T,NV,NE,NR,NC,Fs,∂Fs}
    x::Vector{Vector{T}} # evaluations
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
        @show ex
        (ex == nothing) && continue
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
                Symbolics.build_function(∂exj, parents...,
                    parallel=Symbolics.SerialForm(),
                    checkbounds=true,
                    expression=Val{false})[2])
        end
        push!(Ps, [zeros(T, length(svariables[i]), n) for n in length.(parents)]...)
        push!(Js, [zeros(T, length(svariables[i]), n) for n in length.(parents)]...) # this is wrong we need to figure out a clever way to store intermediate matrices during the backward pass
    end

    # storage vectors
    xs = [zeros(T,n) for n in length.(svariables)]

    # number of roots and children
    num_roots = length(findall(x -> x==nothing, expr))
    num_children = num_vertices - num_roots

    # types
    NV = num_vertices
    NE = length(expr)
    NR = num_roots
    NC = num_children
    types = [T; NV; NE; NR; NC; typeof.([fcts, ∂fcts])]
    signatures = Vector{Vector{Int}}()
    for i = 1:nv(graph)
        ni = indegree(graph, i)
        if ni > 0
            push!(signatures, SVector{1+ni,Int}([i; inneighbors(graph, i)]))
        end
    end
    return SymNet1290{types...}(xs, Ps, Js, graph, fcts, ∂fcts, signatures)
end

findall(x -> x==nothing, [nothing, 1,23,4, nothing])

################################################################################
# evaluation
################################################################################
function evaluation20!(symnet::SymNet1290{T,NV,NE,NR}, xroots::Vector) where {T,NV,NE,NR}
    for i in eachindex(xroots)
        symnet.x[i] .= xroots[i]
    end
    for i in eachindex(symnet.signatures)
        evaluation20!(symnet, i)
    end
end

@generated function evaluation20!(symnet::SymNet1290{T,NV,NE,NR}, i::Int) where {T,NV,NE,NR}
    vec = [:(symnet.x[NR + i]); [:(symnet.x[$j]) for j = 1:NV]]
    return :(symnet.fcts[i]($(vec...)))
end

n = 10
symgraph = generate_symgraph(f1234, [n,n])
signatures0 = symgraph.signatures[1]

xroots0 = [rand(n), rand(n)]

evaluation20!(symgraph, xroots0)
evaluation20!(symgraph, 1)

Main.@code_warntype evaluation20!(symgraph, xroots0)
@benchmark $evaluation20!($symgraph, $xroots0)

norm(symgraph.x[end] - f1234(xroots0...), Inf)
norm(symgraph.x[end], Inf)
norm(f1234(xroots0...), Inf)




# @benchmark $evaluation3!($symgraph, 1, $signatures0)


evaluation3!(symgraph)
Main.@code_warntype evaluation3!(symgraph)
@benchmark $evaluation3!($symgraph)

@generated function evaluation20!(symnet::SymNet1290, d::Int, signature::Vector{Int}, sizeflag::SizeFlag1290{NS}) where NS
    vec = [:(symnet.x[signature[$i]]) for i = 1:NS]
    return :(symnet.fcts[d]($(vec...)))
end

@generated function evaluation3!(symnet::SymNet1290{T,NV,NE,Fs,∂Fs}) where {T,NV,NE,Fs,∂Fs}
    # for (i, signature) in enumerate(symnet.signatures)
    # symnet.signatures[1]

    :(evaluation3!(symnet, $i, symnet.signatures[$i]))
    # return :($vec)
    # evaluation3!(symnet, 2, symnet.signatures[2])
    # end
end

@generated function evaluation3!(symnet::SymNet1290, d::Int, signature::SVector{NS,Int}) where NS
    vec = [:(symnet.x[signature[$i]]) for i = 1:NS]
    return :(symnet.fcts[d]($(vec...)))
end

function evaluation4!(symnet::SymNet1290)
    vec = [:(symnet.x[signature[$i]]) for i = 1:NS]
    return :(symnet.fcts[d]($(vec...)))
end





@generated function ∂g∂ʳposb(mechanism, eqc::EqualityConstraint{T,N,Nc}, body::Body) where {T,N,Nc}
    vec = [:(∂g∂ʳposb(eqc.constraints[$i], getbody(mechanism, eqc.parentid), body, eqc.childids[$i], mechanism.Δt)) for i = 1:Nc]
    return :(vcat($(vec...)))
end

function jacobian!(symnet::SymNet1290{T,NV,NE,Fs,∂Fs}, xin::Vector) where {T,NV,NE,Fs,∂Fs}
    symnet.xouts[1] .= xin
    # evaluation
    for i = 1:N
        symnet.fcts[i](symnet.xouts[i+1], symnet.xouts[i])
    end
    # forward pass
    for i = 1:N
        symnet.∂fcts[i](symnet.Jouts[i], symnet.xouts[i])
    end
    # backward pass
    symnet.Couts[1] .= symnet.Jouts[1]
    for i in 1:N-1
        mul!(symnet.Couts[i+1], symnet.Jouts[i+1], symnet.Couts[i])
    end
    return nothing
end





################################################################################
# test
################################################################################
function f1(x0::Vector)
    x1 = x0 .+ 1
end

function f2(x1::Vector)
    x2 = [x1; sum(x1); log(abs(sum(cos.(x1))))]
end

function f3(x2::Vector)
    x3 = x2.^2 * x2[end] * x2[end-1]
end

function f4(x3::Vector)
    x4 = [sum(x3); sin.(x3 * x3[end])]
end

function f1234(x0::Vector; output_layer::Int=-1, current_layer::Int=0, input_dim::Int=0)
    @layer x0
    x1 = x0 .+ 1
    @layer x1
    x2 = [x1; sum(x1); log(abs(sum(cos.(x1))))]
    @layer x2
    x3 = x2.^2 * x2[end] * x2[end-1]
    @layer x3
    x4 = [sum(x3); sin.(x3 * x3[end])]
    @layer x4
    x5 = x4
    @layer x5
    return x4
end

################################################################################
# example
################################################################################

n = 3
x0 = ones(n)
E0 = f1234(x0)
E1 = f4(f3(f2(f1(x0))))

J0 = FiniteDiff.finite_difference_jacobian(x -> f1234(x), x0)
J1 = FiniteDiff.finite_difference_jacobian(x -> f4(f3(f2(f1(x)))), x0)
@test norm(E0 - E1) < 1e-10
@test norm(J0 - J1) < 1e-10

symnet = generate_symgraph(f1234, n)
monolith_evaluation!(symnet, x0)
monolith_jacobian!(symnet, x0)
@test norm(symnet.xouts[end] - E0) < 1e-5
@test norm(symnet.Couts[end] - J0) < 1e-5
# Main.@code_warntype monolith_evaluation!(symnet, x0)
@benchmark $monolith_evaluation!($symnet, $x0)
@benchmark $monolith_jacobian!($symnet, $x0)

evaluation!(symnet, x0)
jacobian!(symnet, x0)
@test norm(symnet.xouts[end] - E0) < 1e-5
@test norm(symnet.Couts[end] - J0) < 1e-5
# Main.@profiler [eval_fcts!(symnet, x0) for i=1:10000000]
# Main.@code_warntype evaluation!(symnet, x0)
@benchmark $evaluation!($symnet, $x0)
@benchmark $jacobian!($symnet, $x0)


# f1234(x0)
# f1234(x0, output_layer=-1)
# f1234(x0, output_layer=0)
# f1234(x0, output_layer=1)
# f1234(x0, output_layer=2)
# f1234(x0, output_layer=3)
# f1234(x0, output_layer=4)
# f1234(x0, output_layer=5)
# f1234(x0, output_layer=6)
# x0[1]
