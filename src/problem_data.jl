# TODO maybe we could type ProblemData like this ProblemData{T,EX,Eθ,X} where EX = Matrix{T} or SparseMatrixCSC{T,Int}
struct ProblemData{T,X}
    equality_constraint::Vector{T}
    equality_jacobian_variables::Matrix{T}
    equality_jacobian_parameters::Matrix{T}
    equality_jacobian_variables_sparse::SparseMatrixCSC{T,Int}
    equality_jacobian_parameters_sparse::SparseMatrixCSC{T,Int}

    cone_product::Vector{T}
    cone_product_jacobian_duals::Matrix{T}
    cone_product_jacobian_slacks::Matrix{T}
    cone_product_jacobian_duals_sparse::SparseMatrixCSC{T,Int}
    cone_product_jacobian_slacks_sparse::SparseMatrixCSC{T,Int}
    cone_product_jacobian_inverse_dual::Matrix{T}
    cone_product_jacobian_inverse_slack::Matrix{T}
    cone_target::Vector{T}
    custom::X
end

# function ProblemData(num_variables, num_parameters, num_equality, num_cone;
#     custom=nothing, sparse_solver::Bool=false)
#     if sparse_solver
#         return DenseProblemData(num_variables, num_parameters, num_equality, num_cone;
#             custom=custom)
#     else
#         return SparseProblemData(num_variables, num_parameters, num_equality, num_cone;
#             custom=custom)
#     end
# end

function ProblemData(num_variables, num_parameters, num_equality, num_cone;
    custom=nothing)

    equality_constraint = zeros(num_equality)
    equality_jacobian_variables = zeros(num_equality, num_variables)
    equality_jacobian_parameters = zeros(num_equality, num_parameters)
    equality_jacobian_variables_sparse = spzeros(num_equality, num_variables)
    equality_jacobian_parameters_sparse = spzeros(num_equality, num_parameters)

    cone_product = zeros(num_cone)
    cone_product_jacobian_duals = zeros(num_cone, num_cone)
    cone_product_jacobian_slacks = zeros(num_cone, num_cone)
    cone_product_jacobian_duals_sparse = spzeros(num_cone, num_cone)
    cone_product_jacobian_slacks_sparse = spzeros(num_cone, num_cone)
    cone_product_jacobian_inverse_dual = zeros(num_cone, num_cone)
    cone_product_jacobian_inverse_slack = zeros(num_cone, num_cone)
    cone_target = zeros(num_cone)

    ProblemData(
        equality_constraint,
        equality_jacobian_variables,
        equality_jacobian_parameters,
        equality_jacobian_variables_sparse,
        equality_jacobian_parameters_sparse,
        cone_product,
        cone_product_jacobian_duals,
        cone_product_jacobian_slacks,
        cone_product_jacobian_duals_sparse,
        cone_product_jacobian_slacks_sparse,
        cone_product_jacobian_inverse_dual,
        cone_product_jacobian_inverse_slack,
        cone_target,
        custom,
    )
end

# struct SparseProblemData112{T,X} <: ProblemData{T,X}
#     equality_constraint::Vector{T}
#     equality_jacobian_variables::SparseMatrixCSC{T,Int}
#     equality_jacobian_parameters::SparseMatrixCSC{T,Int}
#     cone_product::Vector{T}
#     cone_product_jacobian_duals::Matrix{T}
#     cone_product_jacobian_slacks::Matrix{T}
#     cone_product_jacobian_inverse_dual::Matrix{T}
#     cone_product_jacobian_inverse_slack::Matrix{T}
#     cone_target::Vector{T}
#     custom::X
# end
#
# function SparseProblemData(num_variables, num_parameters, num_equality, num_cone;
#     custom=nothing)
#
#     equality_constraint = zeros(num_equality)
#     equality_jacobian_variables = spzeros(num_equality, num_variables)
#     equality_jacobian_parameters = spzeros(num_equality, num_parameters)
#
#     cone_product = zeros(num_cone)
#     cone_product_jacobian_duals = zeros(num_cone, num_cone)
#     cone_product_jacobian_slacks = zeros(num_cone, num_cone)
#     cone_product_jacobian_inverse_dual = zeros(num_cone, num_cone)
#     cone_product_jacobian_inverse_slack = zeros(num_cone, num_cone)
#     cone_target = zeros(num_cone)
#
#     SparseProblemData112(
#         equality_constraint,
#         equality_jacobian_variables,
#         equality_jacobian_parameters,
#         cone_product,
#         cone_product_jacobian_duals,
#         cone_product_jacobian_slacks,
#         cone_product_jacobian_inverse_dual,
#         cone_product_jacobian_inverse_slack,
#         cone_target,
#         custom,
#     )
# end
