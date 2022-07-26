using Symbolics
using BenchmarkTools

################################################################################
# net
################################################################################
function f1234(x0::Vector)
    x1 = x0 .+ 1
    x2 = [x1; sum(x0); sum(exp.(x0))]
    x3 = x2.^2 * x2[end] * x2[end-1]
    x4 = [sum(x3); sin.(x3 * x3[end])]
end

macro layer11(var)
    return esc(
        quote
            if token < select_layer
                $var .= 0.0
                n = length($var)
                svar = Symbolics.variables(Symbol(:x,token), 1:n)
                $var = svar
            end

            if token == select_layer
                return $var
            end
            token += 1
        end
    )
end

function f1234(x0::Vector; select_layer::Int=-1, token::Int=0)
    @layer11 x0
    x1 = x0 .+ 1
    @layer11 x1
    x2 = [x1; sum(x1); sum(exp.(x1))]
    @layer11 x2
    x3 = x2.^2 * x2[end] * x2[end-1]
    @layer11 x3
    x4 = [sum(x3); sin.(x3 * x3[end])]
    @layer11 x4
    return x4
end

x0 = ones(2)
f1234(x0, select_layer=)


macro assert2(ex, msgs...)
    msg_body = isempty(msgs) ? ex : msgs[1]
    msg = string(msg_body)
    return :($ex ? nothing : throw(AssertionError($msg)))
end



typeof(@macroexpand @layer11)
@macroexpand @layer11


function f12345(x0::Vector; i=4)
    token = 0
    # @out i==1 x1 = x0 .+ 1
    # @out i==2 x2 = [x1; sum(x0); sum(exp.(x0))]
    # @out i==3 x3 = x2.^2 * x2[end] * x2[end-1]
    # @out i==4 x4 = [sum(x3); sin.(x3 * x3[end])]
    x1 = x0 .+ 1
    x2 = [x1; sum(x0); sum(exp.(x0))]
    x3 = x2.^2 * x2[end] * x2[end-1]
    x4 = [sum(x3); sin.(x3 * x3[end])]
    @show_value x4
    @show token
end

macro out(exit_condition, ex)
    :(println($ex))
    # return :($ex)
end

macro output()
    local_token = esc(:(token))
    local_term = esc(:(term))
    ex = :($local_token == $local_term)
    eval(ex)
    # @show typeof(esc(:(token == $variables)))
    # @show typeof(variable)
    # if esc(:(token)) == variable
    #     println(1)
    # end
    # return quote
    #     esc(:(token += 1))
    # end
    # eval(ex)
    # return :($ex)
    # return esc(:(token += 1))
end

function foo(term)
    token = 0
    @output
    @output
    @output
    @output
    @output
    @output
    return token # is zero
end

foo(0)





@macroexpand @output 2



macro show_value(variable)
    quote
        esc(:(token = 1))
        println("The ", $(string(variable)), " you passed is ", $(esc(variable)))
    end
end

f1234(x0)
f12345(x0)


ex = @macroexpand @out true x3 = x2
@show ex

function f1(x0::Vector)
    x1 = x0 .+ 1
end

function f2(x1::Vector)
    x2 = [x1; sum(x0); sum(exp.(x0))]
end

function f3(x2::Vector)
    x3 = x2.^2 * x2[end] * x2[end-1]
end

function f4(x3::Vector)
    x4 = [sum(x3); sin.(x3 * x3[end])]
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

function generate_symnet(n, f, fs)
    xin = zeros(n)
    xv = [Symbolics.variables(:x0, 1:n)]
    ns = [n]
    for i = 1:4
        xin = fs[i](xin)
        push!(ns, length(xin))
        push!(xv, Symbolics.variables(Symbol(:x,i), 1:ns[end]))
    end

    x4 = f(xv[1])
    fct = Symbolics.build_function(x4, xv[1],
        parallel=Symbolics.SerialForm(),
        checkbounds=true,
        expression=Val{false})[2]

    xs = [fs[i](xv[i]) for i=1:4]
    fcts = [Symbolics.build_function(xs[i], xv[i],
                parallel=Symbolics.SerialForm(),
                checkbounds=true,
                expression=Val{false})[2] for i=1:4]

    return SymNet1140(zeros(n), [zeros(ni) for ni in ns], fct, fcts)
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
n = 2
x0 = rand(n)
norm(f1234(x0) - f4(f3(f2(f1(x0))))) < 1e-10

symnet = generate_symnet(n, f1234, [f1,f2,f3,f4])
eval_fct!(symnet, x0)
Main.@code_warntype eval_fct!(symnet, x0)
@benchmark $eval_fct!($symnet, $x0)

eval_fcts!(symnet, x0)
# Main.@profiler [eval_fcts!(symnet, x0) for i=1:1000000]
Main.@code_warntype eval_fcts!(symnet, x0)
@benchmark $eval_fcts!($symnet, $x0)
