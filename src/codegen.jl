function generate_gradients(func::Function, num_variables::Int, num_parameters::Int;
    checkbounds=true,
    threads=false)

    x = Symbolics.variables(:x, 1:num_variables)
    θ = Symbolics.variables(:θ, 1:num_parameters)

    f = num_parameters > 0 ? func(x, θ) : func(x)

    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fx_expr, fθ_expr, fx_sparsity, fθ_sparsity, length(f)
end

# empty_constraint(x) = zeros(0)
# empty_constraint(x, θ) = zeros(0)
