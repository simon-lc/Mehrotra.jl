using Symbolics
using BenchmarkTools

################################################################################
# net
################################################################################

function f1(x0::Vector)
    x1 = x0 .+ 1
end

function f2(x1::Vector)
    x2 = [x1; sum(x1); sum(exp.(x1))]
end

function f3(x2::Vector)
    x3 = x2.^2 * x2[end] * x2[end-1]
end

function f4(x3::Vector)
    x4 = [sum(x3); sin.(x3 * x3[end])]
end

macro layer(var)
    return esc(
        quote
            if token < output_layer
                $var .= 0.0
                input_dim = length($var)
                svar = Symbolics.variables(Symbol(:x, token), 1:input_dim)
                $var = svar
            end

            if token == output_layer
                return $var, input_dim, length($var)
            end
            token += 1
        end
    )
end

function f1234(x0::Vector; output_layer::Int=-1, token::Int=0, input_dim::Int=0)
    @layer x0
    x1 = x0 .+ 1
    @layer x1
    x2 = [x1; sum(x1); sum(exp.(x1))]
    @layer x2
    x3 = x2.^2 * x2[end] * x2[end-1]
    @layer x3
    x4 = [sum(x3); sin.(x3 * x3[end])]
    @layer x4
    return x4
end

function generate_symnet(f::Function, input_dim::Int)
    # storage vectors
    vin = zeros(input_dim)
    vouts = [zeros(input_dim)]

    # single block
    xin = Symbolics.variables(:x, 1:input_dim)
    xout = f(xin)
    fct = Symbolics.build_function(xout, xin,
    parallel=Symbolics.SerialForm(),
    checkbounds=true,
    expression=Val{false})[2]

    # multi-layer
    fcts = []
    # prev_expr = nothing
    i = 1
    while true
        tuple = f(zeros(input_dim), output_layer=i)
        (length(tuple) != 3) && break
        expr, nin, nout = tuple
        push!(vouts, zeros(nout))

        xin = Symbolics.variables(Symbol(:x, i-1), 1:nin)
        push!(fcts,
            Symbolics.build_function(expr, xin,
                parallel=Symbolics.SerialForm(),
                checkbounds=true,
                expression=Val{false})[2])
        i += 1
    end
    return SymNet1140(vin, vouts, fct, fcts)
end


################################################################################
# Symbolics
################################################################################
struct SymNet1140{T,F,Fs}
    xin::Vector{T}
    xouts::Vector{Vector{T}}
    fct::F
    fcts::Fs
end

function eval_fct!(symnet::SymNet1140, xin::Vector)
    symnet.xin .= xin
    symnet.fct(symnet.xouts[end], xin)
    return nothing
end

function eval_fcts!(symnet::SymNet1140{T,F,Fs}, xin::Vector) where {T,F,Fs}
    symnet.xouts[1] .= xin
    for i in eachindex(symnet.fcts)
        symnet.fcts[i](symnet.xouts[i+1], symnet.xouts[i])
    end
    return nothing
end


function eval_f(f::F, xout::Vector{T}, xin::Vector{T}) where {T,F}
    f(xout, xin)
    return nothing
end
################################################################################
# example
################################################################################

n = 15
x0 = rand(n)
norm(f1234(x0) - f4(f3(f2(f1(x0))))) < 1e-10

f1234(x0, output_layer=-1)
f1234(x0, output_layer=0)
f1234(x0, output_layer=1)
f1234(x0, output_layer=2)
f1234(x0, output_layer=3)
f1234(x0, output_layer=4)
f1234(x0, output_layer=5)
f1234(x0, output_layer=6)
f1234(x0)

symnet = generate_symnet(f1234, n)

eval_fct!(symnet, x0)
Main.@code_warntype eval_fct!(symnet, x0)
@benchmark $eval_fct!($symnet, $x0)

eval_fcts!(symnet, x0)
# Main.@profiler [eval_fcts!(symnet, x0) for i=1:10000000]
Main.@code_warntype eval_fcts!(symnet, x0)
@benchmark $eval_fcts!($symnet, $x0)
