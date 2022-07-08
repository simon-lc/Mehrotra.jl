function finite_difference_methods(equality::Function, dim::Dimensions, idx::Indices)

    # in-place evaluation
    function equality_constraint(out, x, θ)
        primals = x[indices.primals]
        duals = x[indices.duals]
        slacks = x[indices.slacks]
        parameters = θ
        out .= equality(primals, duals, slacks, parameters)
    end

    # jacobian variables
    function equality_jacobian_variables(vector_cache, x, θ)
        f(out, x) = equality_constraint(out, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.variables))
        FiniteDiff.finite_difference_jacobian!(matrix_cache, f, x)
        return nothing
    end

    # jacobian parameters
    function equality_jacobian_parameters(vector_cache, x, θ)
        f(out, θ) = equality_constraint(out, x, θ)
        matrix_cache = reshape(vector_cache, (dim.equality, dim.parameters))
        FiniteDiff.finite_difference_jacobian!(matrix_cache, f, θ)
        return nothing
    end

    ex_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.variables)))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(sparse(ones(dim.equality, dim.parameters)))[1:2]...]...))

    methods = ProblemMethods(
        equality_constraint,
        equality_jacobian_variables,
        equality_jacobian_parameters,
        zeros(length(ex_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        eθ_sparsity,
    )
    return methods
end




include("../examples/benchmark_problems/lcp_utils.jl")

num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

As = rand(num_primals, num_primals)
A = As' * As
b = rand(num_primals)
Cs = rand(num_cone, num_cone)
C = Cs * Cs'
d = rand(num_cone)
parameters = [vec(A); b; vec(C); d]

function lcp_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_primals = length(primals)
    num_cone = length(duals)
    A, b, C, d = unpack_lcp_second_order_cone_parameters(parameters, num_primals, num_cone)

    res = [
        A * y + b;
        s - C * z + d;
        # z .* s .- κ[1];
        ]
    return res
end



dimensions = Dimensions(num_primals, num_cone, num_parameters)
indices = Indices(num_primals, num_cone, num_parameters)
finite_difference_methods(lcp_residual, dimensions, indices)


solver = Mehrotra.Solver(nothing, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    methods=finite_difference_methods(lcp_residual, dimensions, indices),
    options=Mehrotra.Options(
        verbose=false,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )
    )

Mehrotra.solve!(solver)
@benchmark $Mehrotra.solve!($solver)




solver = Mehrotra.Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    # methods=FiniteDifferenceMethods(lcp_residual),
    options=Mehrotra.Options(
        verbose=false,
        residual_tolerance=1e-6,
        complementarity_tolerance=1e-6,
        compressed_search_direction=false,
        )
    )

Mehrotra.solve!(solver)
@benchmark $solve!($solver)
