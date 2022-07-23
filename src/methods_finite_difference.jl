function finite_difference_methods(equality::Function, dim::Dimensions, idx::Indices)
    # in-place evaluation
    function equality_constraint(out, x, θ)
        primals = x[idx.primals]function finite_difference_methods(equality::Function, dim::Dimensions, idx::Indices)
    # in-place evaluation
    function equality_constraint(out, x, θ)
        primals = x[idx.primals]
        duals = x[idx.duals]
        slacks = x[idx.slacks]
        parameters = θ
        out .= equality(primals, duals, slacks, parameters)
    end

    # function equality_constraint_compressed(out, x, θ)
    #     primals = x[idx.primals]
    #     duals = x[idx.duals]
    #     slacks = x[idx.slacks]
    #     parameters = θ
    #     out .= equality(primals, duals, slacks, parameters)
    #     D = FiniteDiff.finite_difference_jacobian(
    #         slacks -> equality(primals, duals, slacks, parameters)[idx.slackness], slacks)
    #     Zi = cone_product_jacobian_inverse(slacks, duals, idx_nn, idx_soc)
    #     out[idx.slackness] .-= D * Zi * rs
    # end
    warning = "compressed search direction is not implemented with finite difference methods."
    function equality_constraint_compressed(out, x, θ)
        error(warning)
    end

    # jacobian variables
    function equality_jacobian_variables(vector_cache, x, θ)
        f(out, x) = equality_constraint(out, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        FiniteDiff.finite_difference_jacobian!(matrix_cache, f, x)
        return nothing
    end

    # jacobian variables compressed
    function equality_jacobian_variables_compressed(vector_cache, x, θ)
        error(warning)
    end
    # # jacobian variables compressed
    # function equality_jacobian_variables_compressed(vector_cache, x, θ)
    #     f(out, x) = equality_constraint(out, x, θ)
    #     matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
    #     FiniteDiff.finite_difference_jacobian!(matrix_cache, f, x)
    #     D = FiniteDiff.finite_difference_jacobian(
    #         slacks -> equality(primals, duals, slacks, parameters)[idx.slackness], slacks)
    #     S = cone_product_jacobian(duals, slacks, idx_nn, idx_soc)
    #     Zi = cone_product_jacobian_inverse(slacks, duals, idx_nn, idx_soc)
    #     matrix[idx.slackness, idx.duals] .-= D * Zi * S
    #     return nothing
    # end

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

    ex_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    exc_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.equality)))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))

    methods = ProblemMethods(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        correction,
        correction_compressed,
        slack_direction,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
    )

    return methods
end

        duals = x[idx.duals]
        slacks = x[idx.slacks]
        parameters = θ
        out .= equality(primals, duals, slacks, parameters)
    end

    # function equality_constraint_compressed(out, x, θ)
    #     primals = x[idx.primals]
    #     duals = x[idx.duals]
    #     slacks = x[idx.slacks]
    #     parameters = θ
    #     out .= equality(primals, duals, slacks, parameters)
    #     D = FiniteDiff.finite_difference_jacobian(
    #         slacks -> equality(primals, duals, slacks, parameters)[idx.slackness], slacks)
    #     Zi = cone_product_jacobian_inverse(slacks, duals, idx_nn, idx_soc)
    #     out[idx.slackness] .-= D * Zi * rs
    # end
    warning = "compressed search direction is not implemented with finite difference methods."
    function equality_constraint_compressed(out, x, θ)
        error(warning)
    end

    # jacobian variables
    function equality_jacobian_variables(vector_cache, x, θ)
        f(out, x) = equality_constraint(out, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        FiniteDiff.finite_difference_jacobian!(matrix_cache, f, x)
        return nothing
    end

    # jacobian variables compressed
    function equality_jacobian_variables_compressed(vector_cache, x, θ)
        error(warning)
    end
    # # jacobian variables compressed
    # function equality_jacobian_variables_compressed(vector_cache, x, θ)
    #     f(out, x) = equality_constraint(out, x, θ)
    #     matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
    #     FiniteDiff.finite_difference_jacobian!(matrix_cache, f, x)
    #     D = FiniteDiff.finite_difference_jacobian(
    #         slacks -> equality(primals, duals, slacks, parameters)[idx.slackness], slacks)
    #     S = cone_product_jacobian(duals, slacks, idx_nn, idx_soc)
    #     Zi = cone_product_jacobian_inverse(slacks, duals, idx_nn, idx_soc)
    #     matrix[idx.slackness, idx.duals] .-= D * Zi * S
    #     return nothing
    # end

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

    ex_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    exc_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.equality)))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))

    methods = ProblemMethods(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        correction,
        correction_compressed,
        slack_direction,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
    )

    return methods
end
