# TODO maybe we could type ProblemData like this ProblemData{T,EX,Eθ,X} where EX = Matrix{T} or SparseMatrixCSC{T,Int}
struct ProblemData{T,X}
    equality_constraint::Vector{T} #e
    equality_constraint_compressed::Vector{T} #ec
    equality_jacobian_variables::SparseMatrixCSC{T,Int} #ex
    equality_jacobian_variables_compressed::SparseMatrixCSC{T,Int} #exc
    equality_jacobian_parameters::SparseMatrixCSC{T,Int} #eθ

    cone_product::Vector{T} #s∘z
    cone_product_jacobian_duals::SparseMatrixCSC{T,Int} #S
    cone_product_jacobian_slacks::SparseMatrixCSC{T,Int} #Z
    custom::X
end

function ProblemData(num_variables, num_parameters, num_equality, num_cone;
    custom=nothing)

    equality_constraint = zeros(num_equality)
    equality_constraint_compressed = zeros(num_equality)
    equality_jacobian_variables = spzeros(num_equality, num_variables)
    equality_jacobian_variables_compressed = spzeros(num_equality, num_equality)
    equality_jacobian_parameters = spzeros(num_equality, num_parameters)

    cone_product = zeros(num_cone)
    cone_product_jacobian_duals = spzeros(num_cone, num_cone)
    cone_product_jacobian_slacks = spzeros(num_cone, num_cone)

    ProblemData(
        equality_constraint,
        equality_constraint_compressed,
        equality_jacobian_variables,
        equality_jacobian_variables_compressed,
        equality_jacobian_parameters,
        cone_product,
        cone_product_jacobian_duals,
        cone_product_jacobian_slacks,
        custom,
    )
end
