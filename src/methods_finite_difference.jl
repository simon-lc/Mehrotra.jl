function finite_difference_methods(equality::Function, dim::Dimensions, idx::Indices;
        regularizer=1e-6)
    parameter_keywords = idx.parameter_keywords

    # in-place evaluation
    function equality_constraint(out, x, θ)
        primals = x[idx.primals]
        duals = x[idx.duals]
        slacks = x[idx.slacks]
        parameters = θ
        out .= equality(primals, duals, slacks, parameters)
    end

    warning = "compressed search direction is not implemented with finite difference methods."
    function equality_constraint_compressed(out, x, θ)
        @warn warning
    end

    # jacobian variables
    function equality_jacobian_variables(vector_cache, x, θ)
        f(out, x) = equality_constraint(out, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        FiniteDiff.finite_difference_jacobian!(matrix_cache, f, x)
        matrix_cache[idx.primals, idx.primals] .+= regularizer * Diagonal(ones(dim.primals))
        matrix_cache[idx.duals, idx.duals] .-= regularizer * Diagonal(ones(dim.duals))
        return nothing
    end

    # jacobian variables compressed
    function equality_jacobian_variables_compressed(vector_cache, x, θ)
        error(warning)
    end

    # correction
    function correction(c, r, Δza, Δsa, κ)
        c .= r
        c[idx.cone_product] .-= (κ - cone_product(Δza, Δsa, idx.cone_nonnegative, idx.cone_second_order))
        return nothing
    end

    # correction compressed
    function correction_compressed(cc, r, Δza, Δsa, x, κ)
        error(warning)
    end

    # slack direction
    function slack_direction(Δs, Δz, x, rs)
        error(warning)
    end

    # jacobian parameters
    function equality_jacobian_parameters(vector_cache, x, θ)
        f(out, θ) = equality_constraint(out, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.parameters))
        FiniteDiff.finite_difference_jacobian!(matrix_cache, f, θ)
        return nothing
    end

    equality_jacobian_keywords = Vector{Function}()
    for k in eachindex(parameter_keywords)
        function func(vector_cache, x, θ)
            function f(out, θi)
                θc = copy(θ)
                θc[parameter_keywords[k]] .= θi
                equality_constraint(out, x, θc)
            end
            matrix_cache = reshape(vector_cache, (dim.equality, length(parameter_keywords[k])))
            FiniteDiff.finite_difference_jacobian!(matrix_cache, f, θ[parameter_keywords[k]])
            return nothing
        end
        push!(equality_jacobian_keywords, func)
    end

    ex_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    exc_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.equality)))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))
    eθ_indices = zeros(Int, dim.equality, dim.parameters)
    vec(eθ_indices) .= 1:length(eθ_indices)
    ek_indices = Vector{Vector{Int}}()
    for (key, val) in parameter_keywords
        indices = vec(eθ_indices[:,val])
        push!(ek_indices, indices)
    end

    methods = ProblemMethods(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        equality_jacobian_keywords,
        correction,
        correction_compressed,
        slack_direction,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
        ek_indices,
    )

    return methods
end
