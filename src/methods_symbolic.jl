abstract type AbstractProblemMethods{T,E,EC,EX,EXC,EP,C,CC,S}
end

struct ProblemMethods{T,E,EC,EX,EXC,EP,C,CC,S} <: AbstractProblemMethods{T,E,EC,EX,EXC,EP,C,CC,S}
    equality_constraint::E                             # e
    equality_constraint_compressed::EC                 # ec
    equality_jacobian_variables::EX                    # ex
    equality_jacobian_variables_compressed::EXC        # exc
    equality_jacobian_parameters::EP                   # eθ
    equality_jacobian_keywords#::EK                     # ek
    correction::C                                      # c
    correction_compressed::CC                          # cc
    slack_direction::S                                 # s
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_variables_compressed_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_keywords_cache::Vector{Vector{T}}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_variables_compressed_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_keywords_sparsity::Vector{Vector{Tuple{Int,Int}}}
end

function symbolics_methods(equality::Function, dim::Dimensions, idx::Indices;
    parameter_keywords=Dict{Symbol,Vector{Int}}(:all => idx.parameters),
    )
    e, ec, ex, exc, eθ, ek, c, cc, s, ex_sparsity, exc_sparsity, eθ_sparsity, ek_sparsity = generate_symbolic_gradients(equality, dim, idx)

    methods = ProblemMethods(
        e,
        ec,
        ex,
        exc,
        eθ,
        ek,
        c,
        cc,
        s,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        [zeros(length(s)) for s in ek_sparsity],
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
        ek_sparsity,
    )

    return methods
end

function generate_symbolic_gradients(func::Function, dim::Dimensions, ind::Indices;
        parameter_keywords=Dict{Symbol,Vector{Int}}(:all => ind.parameters),
        checkbounds=true,
        threads=false)

    parallel = threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()
    parallel_parameters = (threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()

    idx_nn = ind.cone_nonnegative
    idx_soc = ind.cone_second_order

    x = Symbolics.variables(:x, 1:dim.variables) # variables
    θ = Symbolics.variables(:θ, 1:dim.parameters) # parameters
    κ = Symbolics.variables(:κ, 1:dim.cone) # central path
    r = Symbolics.variables(:r, 1:dim.variables) # residual
    rs = Symbolics.variables(:rs, 1:dim.cone) # cone product residual with corrections
    Δz = Symbolics.variables(:Δz, 1:dim.cone) # dual step
    Δza = Symbolics.variables(:Δza, 1:dim.cone) # dual affine step
    Δsa = Symbolics.variables(:Δsa, 1:dim.cone) # slack affine step
    y = x[ind.primals]
    z = x[ind.duals]
    s = x[ind.slacks]

    # equality residual
    f = dim.parameters > 0 ?
        func(y, z, s, θ) :
        func(y, z, s)

    # equality jacobians
    fx = Symbolics.sparsejacobian(f, x)
    fθ = Symbolics.sparsejacobian(f, θ)
    fk = [Symbolics.sparsejacobian(f, θ[parameter_keywords[k]]) for k in eachindex(parameter_keywords)]

    # compressed search direction
    D = fx[ind.slackness, ind.slacks]
    Zi = cone_product_jacobian_inverse(s, z, idx_nn, idx_soc)
    S = cone_product_jacobian(z, s, idx_nn, idx_soc)

    # compressed equality residual
    fc = copy(f)
    fc[ind.slackness] .-= D * Zi * cone_product(s, z, idx_nn, idx_soc)

    # compressed equality jacobians
    fxc = copy(fx[ind.equality, [ind.primals; ind.duals]])
    fxc[ind.slackness, ind.duals] .-= D * Zi * S
    fxc = sparse(fxc)

    # correction
    c = copy(r)
    c[ind.cone_product] .-= (κ - cone_product(Δza, Δsa, idx_nn, idx_soc))

    # correction compressed
    cc = copy(r)
    cc[ind.slackness] .+= D * Zi * (κ - cone_product(Δza, Δsa, idx_nn, idx_soc))
    # cc[ind.cone_product] .-= κ

    # slack direction
    Δs = Zi * (-rs - S * Δz)

    # sparsity
    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fxc_sparsity = collect(zip([findnz(fxc)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))
    fk_sparsity = [collect(zip([findnz(fki)[1:2]...]...)) for fki in fk]

    # expressions
    f_expr = Symbolics.build_function(f, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fc_expr = Symbolics.build_function(fc, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fxc_expr = Symbolics.build_function(fxc.nzval, x, θ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fk_expr = [Symbolics.build_function(fki.nzval, x, θ,
        parallel=parallel_parameters,
        checkbounds=checkbounds,
        expression=Val{false})[2] for fki in fk]
    c_expr = Symbolics.build_function(c, r, Δza, Δsa, κ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    cc_expr = Symbolics.build_function(cc, r, Δza, Δsa, x, κ,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]
    s_expr = Symbolics.build_function(Δs, Δz, x, rs,
        parallel=parallel,
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return f_expr, fc_expr, fx_expr, fxc_expr, fθ_expr, fk_expr, c_expr, cc_expr, s_expr, fx_sparsity, fxc_sparsity, fθ_sparsity, fk_sparsity
end
