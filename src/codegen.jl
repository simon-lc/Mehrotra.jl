function generate_gradients(func::Function, dim::Dimensions, ind::Indices;
        checkbounds=true,
        threads=false)

    parallel = threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    parallel_parameters = (threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()

    x = Symbolics.variables(:x, 1:dim.variables)
    θ = Symbolics.variables(:θ, 1:dim.parameters)

    f = dim.parameters > 0 ?
        func(x[ind.primals], x[ind.duals], x[ind.slacks], θ) :
        func(x[ind.primals], x[ind.duals], x[ind.slacks])

    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))

    f_expr = Symbolics.build_function(f, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fx_expr, fθ_expr, fx_sparsity, fθ_sparsity
end

function generate_structured_gradients(func::Function, dim::Dimensions, ind::Indices;
        checkbounds=true,
        threads=false)

    parallel = threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    parallel_parameters = (threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()

    x = Symbolics.variables(:x, 1:dim.variables)
    θ = Symbolics.variables(:θ, 1:dim.parameters)

    f = dim.parameters > 0 ?
        func(x[ind.primals], x[ind.duals], x[ind.slacks], θ) :
        func(x[ind.primals], x[ind.duals], x[ind.slacks])
    optimality = f[ind.optimality]
    slackness = f[ind.slackness]

    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)
    optimality_y = fx[ind.optimality, ind.primals]
    optimality_z = fx[ind.optimality, ind.duals]
    optimality_θ = fθ[ind.optimality, :]
    slackness_y = fx[ind.slackness, ind.primals]
    slackness_s = fx[ind.slackness, ind.slacks]
    slackness_θ = fθ[ind.slackness, :]

    oy_sparsity = collect(zip([findnz(optimality_y)[1:2]...]...))
    oz_sparsity = collect(zip([findnz(optimality_z)[1:2]...]...))
    oθ_sparsity = collect(zip([findnz(optimality_θ)[1:2]...]...))

    sy_sparsity = collect(zip([findnz(slackness_y)[1:2]...]...))
    ss_sparsity = collect(zip([findnz(slackness_s)[1:2]...]...))
    sθ_sparsity = collect(zip([findnz(slackness_θ)[1:2]...]...))

    optimality_y = fx[ind.optimality, ind.primals]
    optimality_z = fx[ind.optimality, ind.duals]
    optimality_θ = fθ[ind.optimality, :]
    slackness_y = fx[ind.slackness, ind.primals]
    slackness_s = fx[ind.slackness, ind.slacks]
    slackness_θ = fθ[ind.slackness, :]

    o_expr = Symbolics.build_function(optimality, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    oy_expr = Symbolics.build_function(optimality_y.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    oz_expr = Symbolics.build_function(optimality_z.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    oθ_expr = Symbolics.build_function(optimality_θ.nzval, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2]

    s_expr = Symbolics.build_function(slackness, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    sy_expr = Symbolics.build_function(slackness_y.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    ss_expr = Symbolics.build_function(slackness_s.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    sθ_expr = Symbolics.build_function(slackness_θ.nzval, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return o_expr,
        oy_expr,
        oz_expr,
        oθ_expr,
        s_expr,
        sy_expr,
        ss_expr,
        sθ_expr,
        oy_sparsity,
        oz_sparsity,
        oθ_sparsity,
        sy_sparsity,
        ss_sparsity,
        sθ_sparsity
end




# empty_constraint(x) = zeros(0)
# empty_constraint(x, θ) = zeros(0)
