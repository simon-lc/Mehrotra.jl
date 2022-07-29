using Symbolics
using BenchmarkTools
using LinearAlgebra
using Test
using Random


################################################################################
# macro
################################################################################
"""
    This macro splits the function into layers and pass variables as input to a layer
    and compute the output of the layer. This generate simple exparessions.
"""
macro layer(var)
    return esc(
        quote
            if current_layer < output_layer
                $var .= 0.0
                input_dim = length($var)
                svar = Symbolics.variables(Symbol(:x, current_layer), 1:input_dim)
                $var = svar
            end

            if current_layer == output_layer
                return $var, input_dim, length($var)
            end
            current_layer += 1
        end
    )
end

################################################################################
# net
################################################################################
struct SymNet1180{T,N,F,Fs,∂F,∂Fs}
    xouts::Vector{Vector{T}}
    Jouts::Vector{Matrix{T}}
    Couts::Vector{Matrix{T}}
    fct::F
    fcts::Fs
    ∂fct::∂F
    ∂fcts::∂Fs
end

function generate_symnet(f::Function, input_dim::Int; T=Float64)
    # storage vectors
    xouts = [zeros(T,input_dim)]
    Jouts = Vector{Matrix{T}}()
    Couts = Vector{Matrix{T}}()

    # single block
    xin = Symbolics.variables(:x0, 1:input_dim)
    expr = f(xin)
    fct = Symbolics.build_function(expr, xin,
        parallel=Symbolics.SerialForm(),
        checkbounds=true,
        expression=Val{false})[2]
    # ∂expr = Symbolics.sparsejacobian(expr, xin)
    ∂expr = Symbolics.jacobian(expr, xin)
    ∂fct = Symbolics.build_function(∂expr, xin,
        parallel=Symbolics.SerialForm(),
        checkbounds=true,
        expression=Val{false})[2]

    # multi-layer
    fcts = []
    ∂fcts = []
    i = 1
    while true
        tuple = f(zeros(input_dim), output_layer=i)
        (length(tuple) != 3) && break # we observe that the out is a 3-tuple only for a valid layer.
        expr, nin, nout = tuple
        push!(xouts, zeros(nout))
        push!(Jouts, zeros(nout, nin))
        push!(Couts, zeros(nout, input_dim))

        xin = Symbolics.variables(Symbol(:x, i-1), 1:nin)
        # ∂expr = Symbolics.sparsejacobian(expr, xin)
        ∂expr = Symbolics.jacobian(expr, xin)
        push!(fcts,
            Symbolics.build_function(expr, xin,
                parallel=Symbolics.SerialForm(),
                checkbounds=true,
                expression=Val{false})[2])
        push!(∂fcts,
            Symbolics.build_function(∂expr, xin,
                parallel=Symbolics.SerialForm(),
                checkbounds=true,
                expression=Val{false})[2])

        i += 1
    end
    # types
    N = length(fcts)
    types = [T; N; typeof.([fct, fcts, ∂fct, ∂fcts])]
    return SymNet1180{types...}(xouts, Jouts, Couts, fct, fcts, ∂fct, ∂fcts)
end

################################################################################
# evaluation
################################################################################
function monolith_evaluation!(symnet::SymNet1180, xin::Vector)
    symnet.xouts[1] .= xin
    symnet.fct(symnet.xouts[end], symnet.xouts[1])
    return nothing
end

function monolith_jacobian!(symnet::SymNet1180, xin::Vector)
    symnet.xouts[1] .= xin
    symnet.∂fct(symnet.Couts[end], symnet.xouts[1])
    return nothing
end

function evaluation!(symnet::SymNet1180{T,N,F,Fs}, xin::Vector) where {T,N,F,Fs}
    symnet.xouts[1] .= xin
    for i in eachindex(symnet.fcts)
        symnet.fcts[i](symnet.xouts[i+1], symnet.xouts[i])
    end
    return nothing
end

function jacobian!(symnet::SymNet1180{T,N,F,Fs}, xin::Vector) where {T,N,F,Fs}
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
