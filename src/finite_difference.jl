function finite_difference_methods(equality::Function, dim::Dimensions, idx::Indices)
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
        @warn warning
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
        @warn warning
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
        @warn warning
    end

    # slack direction
    function slack_direction(Δs, Δz, x, rs)
        @warn warning
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



#
# include("../examples/benchmark_problems/lcp_utils.jl")
#
# num_primals = 10
# num_cone = 10
# num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone
#
# idx_nn = collect(1:num_cone)
# idx_soc = [collect(1:0)]
#
# As = rand(num_primals, num_primals)
# A = As' * As
# B = rand(num_primals, num_cone)
# C = B'
# d = rand(num_primals)
# e = zeros(num_cone)
# parameters = [vec(A); vec(B); vec(C); d; e]
#
# function lcp_residual(primals, duals, slacks, parameters)
#     y, z, s = primals, duals, slacks
#     num_primals = length(primals)
#     num_cone = length(duals)
#     A, b, C, d = unpack_lcp_second_order_cone_parameters(parameters, num_primals, num_cone)
#
#     res = [
#         A * y + b;
#         s - C * z + d;
#         # z .* s .- κ[1];
#         ]
#     return res
# end
#
#
#
# dimensions = Dimensions(num_primals, num_cone, num_parameters)
# indices = Indices(num_primals, num_cone, num_parameters)
# finite_difference_methods(lcp_residual, dimensions, indices)
#
#
# solver = Mehrotra.Solver(nothing, num_primals, num_cone,
#     parameters=parameters,
#     nonnegative_indices=idx_nn,
#     second_order_indices=idx_soc,
#     methods=finite_difference_methods(lcp_residual, dimensions, indices),
#     options=Mehrotra.Options(
#         verbose=false,
#         residual_tolerance=1e-6,
#         complementarity_tolerance=1e-6,
#         compressed_search_direction=false,
#         )
#     )
#
# Mehrotra.solve!(solver)
# # @benchmark $Mehrotra.solve!($solver)
# solver.trace.iterations
#
#
#
# solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
#     parameters=parameters,
#     nonnegative_indices=idx_nn,
#     second_order_indices=idx_soc,
#     # methods=FiniteDifferenceMethods(lcp_residual),
#     options=Mehrotra.Options(
#         verbose=false,
#         residual_tolerance=1e-6,
#         complementarity_tolerance=1e-6,
#         compressed_search_direction=false,
#         )
#     )
#
# Mehrotra.solve!(solver)
# # @benchmark $solve!($solver)
# solver.trace.iterations
#
