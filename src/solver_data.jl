struct SolverData{T}
    residual::Point{T}
    compressed_residual::Point{T}
    jacobian_variables::SparseMatrixCSC{T,Int}
    jacobian_variables_sparse::BlockSparse116{T}
    dense_jacobian_variables::Matrix{T}
    compressed_jacobian_variables::SparseMatrixCSC{T,Int}
    dense_compressed_jacobian_variables::Matrix{T}
    cone_product_jacobian_inverse_slack::Matrix{T}
    cone_product_jacobian_duals::Matrix{T}
    cone_product_jacobian_ratio::Matrix{T}
    jacobian_parameters::SparseMatrixCSC{T,Int}
    jacobian_parameters_sparse::BlockSparse116{T}
    step::Point{T}
    step_correction::Point{T}
    point_temporary::Point{T}
    merit::Vector{T}
    merit_gradient::Vector{T}
    constraint_violation::Vector{T}
    solution_sensitivity::Matrix{T}
end

function SolverData(dim::Dimensions, idx::Indices, p_data::ProblemData;
    T=Float64)

    num_variables = dim.variables
    num_parameters = dim.parameters
    num_equality = dim.equality
    num_cone = dim.cone

    residual = Point(dim, idx)
    compressed_residual = Point(dim, idx)

    blocks = [
        p_data.equality_jacobian_variables_sparse,
        p_data.cone_product_jacobian_duals_sparse,
        p_data.cone_product_jacobian_slacks_sparse,
        ]
    ranges = [(idx.equality, idx.variables), (idx.cone_product, idx.duals), (idx.cone_product, idx.slacks)]
    names = [:equality_jacobian_variables, :cone_jacobian_duals, :cone_jacobian_slacks]
    jacobian_variables = spzeros(num_variables, num_variables)
    jacobian_variables_sparse = BlockSparse116(num_variables, num_variables, blocks, ranges, names=names)

    dense_jacobian_variables = zeros(num_variables, num_variables)
    compressed_jacobian_variables = spzeros(num_equality, num_equality)
    dense_compressed_jacobian_variables = zeros(num_equality, num_equality)

    cone_product_jacobian_inverse_slack = zeros(num_cone, num_cone)
    cone_product_jacobian_duals = zeros(num_cone, num_cone)
    cone_product_jacobian_ratio = zeros(num_cone, num_cone)

    blocks = [p_data.equality_jacobian_parameters_sparse]
    ranges = [(idx.equality, idx.parameters)]
    names = [:equality_jacobian_parameters]
    jacobian_parameters = spzeros(num_variables, num_parameters)
    jacobian_parameters_sparse = BlockSparse116(num_variables, num_parameters, blocks, ranges, names=names)

    step = Point(dim, idx)
    step_correction = Point(dim, idx)
    point_temporary = Point(dim, idx)

    merit = zeros(1)
    merit_gradient = zeros(num_variables)

    constraint_violation = zeros(num_variables)

    solution_sensitivity = zeros(num_variables, num_parameters)

    SolverData(
        residual,
        compressed_residual,
        jacobian_variables,
        jacobian_variables_sparse,
        dense_jacobian_variables,
        compressed_jacobian_variables,
        dense_compressed_jacobian_variables,
        cone_product_jacobian_inverse_slack,
        cone_product_jacobian_duals,
        cone_product_jacobian_ratio,
        jacobian_parameters,
        jacobian_parameters_sparse,
        step,
        step_correction,
        point_temporary,
        merit,
        merit_gradient,
        constraint_violation,
        solution_sensitivity,
    )
end
