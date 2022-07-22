struct SolverData{T}
    residual::Point{T}
    residual_compressed::Point{T}
    jacobian_variables_dense::Matrix{T}
    jacobian_variables_sparse::BlockSparse{T}
    jacobian_variables_compressed_dense::Matrix{T}
    jacobian_variables_compressed_sparse::SparseMatrixCSC{T,Int}
    # slackness_jacobian_slacks::Vector{T}
    # cone_product_jacobian_inverse_slack::Matrix{T}
    # cone_product_jacobian_duals::Matrix{T}
    # cone_product_jacobian_ratio::Matrix{T}
    jacobian_parameters::SparseMatrixCSC{T,Int}
    jacobian_parameters_sparse::BlockSparse{T}
    step::Point{T}
    step_correction::Point{T}
    # point_temporary::Point{T}
    # merit::Vector{T}
    # merit_gradient::Vector{T}
    # constraint_violation::Vector{T}
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
    # jacobian_variables_sparse = spzeros(num_variables, num_variables)

    jacobian_variables_compressed_dense = zeros(num_equality, num_equality)
    jacobian_variables_compressed_sparse = spzeros(num_equality, num_equality)

    # slackness_jacobian_slacks = zeros(num_cone)

    # cone_product_jacobian_inverse_slack = zeros(num_cone, num_cone)
    # cone_product_jacobian_duals = zeros(num_cone, num_cone)
    # cone_product_jacobian_ratio = zeros(num_cone, num_cone)

    blocks = [p_data.equality_jacobian_parameters]
    ranges = [(idx.equality, idx.parameters)]
    names = [:equality_jacobian_parameters]
    jacobian_parameters = spzeros(num_variables, num_parameters)
    jacobian_parameters_sparse = BlockSparse(num_variables, num_parameters, blocks, ranges, names=names)
    # jacobian_parameters_sparse = spzeros(num_variables, num_parameters)

    step = Point(dim, idx)
    step_correction = Point(dim, idx)
    # point_temporary = Point(dim, idx)

    # merit = zeros(1)
    # merit_gradient = zeros(num_variables)

    # constraint_violation = zeros(num_variables)

    solution_sensitivity = zeros(num_variables, num_parameters)

    SolverData(
        residual,
        residual_compressed,
        jacobian_variables_dense,
        jacobian_variables_sparse,
        jacobian_variables_compressed_dense,
        jacobian_variables_compressed_sparse,
        # slackness_jacobian_slacks,
        # cone_product_jacobian_inverse_slack,
        # cone_product_jacobian_duals,
        # cone_product_jacobian_ratio,
        jacobian_parameters,
        jacobian_parameters_sparse,
        step,
        step_correction,
        # point_temporary,
        # merit,
        # merit_gradient,
        # constraint_violation,
        solution_sensitivity,
    )
end
