# TODO maybe we could type ProblemData like this ProblemData{T,EX,Eθ,X} where EX = Matrix{T} or SparseMatrixCSC{T,Int}
struct ProblemData{T,X}
    equality_constraint::Vector{T} #e
    equality_constraint_compressed::Vector{T} #ec
    equality_jacobian_variables_sparse::SparseMatrixCSC{T,Int} #ex
    equality_jacobian_variables_compressed_sparse::SparseMatrixCSC{T,Int} #exc
    equality_jacobian_parameters_sparse::SparseMatrixCSC{T,Int} #eθ

    cone_product::Vector{T} #s∘Z
    cone_product_jacobian_duals_sparse::SparseMatrixCSC{T,Int} #S
    cone_product_jacobian_slacks_sparse::SparseMatrixCSC{T,Int} #Z
    cone_product_jacobian_inverse_duals_sparse::SparseMatrixCSC{T,Int} #Si
    cone_product_jacobian_inverse_slacks_sparse::SparseMatrixCSC{T,Int} #Zi
    cone_target::Vector{T} 
    custom::X
end

function ProblemData(num_variables, num_parameters, num_equality, num_cone;
    custom=nothing)

    equality_constraint = zeros(num_equality)
    equality_constraint_compressed = zeros(num_equality)
    equality_jacobian_variables_sparse = spzeros(num_equality, num_variables)
    equality_jacobian_variables_compressed_sparse = spzeros(num_equality, num_equality)
    equality_jacobian_parameters_sparse = spzeros(num_equality, num_parameters)

    cone_product = zeros(num_cone)
    cone_product_jacobian_duals_sparse = spzeros(num_cone, num_cone)
    cone_product_jacobian_slacks_sparse = spzeros(num_cone, num_cone)
    cone_product_jacobian_inverse_duals_sparse = spzeros(num_cone, num_cone)
    cone_product_jacobian_inverse_slacks_sparse = spzeros(num_cone, num_cone)
    cone_target = zeros(num_cone)

    ProblemData(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables_sparse,
        equality_jacobian_variables_compressed_sparse,
        equality_jacobian_parameters_sparse,
        cone_product,
        cone_product_jacobian_duals_sparse,
        cone_product_jacobian_slacks_sparse,
        cone_product_jacobian_inverse_duals_sparse,
        cone_product_jacobian_inverse_slacks_sparse,
        cone_target,
        custom,
    )
end
