struct SolverData{T}
    residual::Point{T}
    residual_compressed::Point{T}
    jacobian_variables_dense::Matrix{T}
    jacobian_variables_sparse::BlockSparse{T}
    jacobian_variables_compressed_dense::Matrix{T}
    jacobian_variables_compressed_sparse::SparseMatrixCSC{T,Int}
    jacobian_parameters::SparseMatrixCSC{T,Int}
    step::Point{T}
    step_correction::Point{T}
    solution_sensitivity::Matrix{T}
end

function SolverData(dim::Dimensions, idx::Indices, p_data::ProblemData;
    T=Float64)

    num_variables = dim.variables
    num_parameters = dim.parameters
    num_equality = dim.equality
    num_cone = dim.cone

    residual = Point(dim, idx)
    residual_compressed = Point(dim, idx)

    jacobian_variables_dense = zeros(num_variables, num_variables)
    blocks = [
        p_data.equality_jacobian_variables,
        p_data.cone_product_jacobian_duals,
        p_data.cone_product_jacobian_slacks,
        ]
    ranges = [(idx.equality, idx.variables), (idx.cone_product, idx.duals), (idx.cone_product, idx.slacks)]
    names = [:equality_jacobian_variables, :cone_jacobian_duals, :cone_jacobian_slacks]
    jacobian_variables_sparse = BlockSparse(num_variables, num_variables, blocks, ranges, names=names)

    jacobian_variables_compressed_dense = zeros(num_equality, num_equality)
    jacobian_variables_compressed_sparse = spzeros(num_equality, num_equality)

    jacobian_parameters =spzeros(num_variables, num_parameters)

    step = Point(dim, idx)
    step_correction = Point(dim, idx)


    solution_sensitivity = zeros(num_variables, num_parameters)

    SolverData(
        residual,
        residual_compressed,
        jacobian_variables_dense,
        jacobian_variables_sparse,
        jacobian_variables_compressed_dense,
        jacobian_variables_compressed_sparse,
        jacobian_parameters,
        step,
        step_correction,
        solution_sensitivity,
    )
end
