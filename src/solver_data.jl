struct SolverData{T}
    residual::Point{T}
    compressed_residual::Point{T}
    jacobian_variables::BlockSparse116{T}
    dense_jacobian_variables::Matrix{T}
    compressed_jacobian_variables::SparseMatrixCSC{T,Int}
    dense_compressed_jacobian_variables::Matrix{T}
    cone_product_jacobian_inverse_slack::Matrix{T}
    cone_product_jacobian_dual::Matrix{T}
    cone_product_jacobian_ratio::Matrix{T}
    jacobian_parameters::BlockSparse116{T}
    step::Point{T}
    step_correction::Point{T}
    point_temporary::Point{T}
    merit::Vector{T}
    merit_gradient::Vector{T}
    constraint_violation::Vector{T}
    solution_sensitivity::Matrix{T}
end

function SolverData(dim::Dimensions, idx::Indices;
    T=Float64)

    num_variables = dim.variables
    num_parameters = dim.parameters
    num_equality = dim.equality
    num_cone = dim.cone

    residual = Point(dim, idx)
    compressed_residual = Point(dim, idx)

    names = [:equality_jacobian_variables, :cone_jacobian_duals, :cone_jacobian_slacks]
    blocks = [equality_jacobian_variables, cone_jacobian_duals, cone_jacobian_slacks]
    ranges = [(idx.equality, idx.variables), (idx.cone_product, idx.duals), (idx.cone_product, idx.slacks)]
    jacobian_variables = BlockSparse116(num_variables, num_variables, blocks, ranges, names)

    dense_jacobian_variables = zeros(num_variables, num_variables)
    compressed_jacobian_variables = spzeros(num_equality, num_equality)
    dense_compressed_jacobian_variables = zeros(num_equality, num_equality)

    cone_product_jacobian_inverse_slack = zeros(num_cone, num_cone)
    cone_product_jacobian_dual = zeros(num_cone, num_cone)
    cone_product_jacobian_ratio = zeros(num_cone, num_cone)

    blocks = [equality_jacobian_parameters]
    ranges = [(idx.equality, idx.variables)]
    names = [:equality_jacobian_parameters]
    jacobian_parameters = BlockSparse116(num_variables, num_parameters, blocks, ranges, names)

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
        dense_jacobian_variables,
        compressed_jacobian_variables,
        dense_compressed_jacobian_variables,
        cone_product_jacobian_inverse_slack,
        cone_product_jacobian_dual,
        cone_product_jacobian_ratio,
        jacobian_parameters,
        step,
        step_correction,
        point_temporary,
        merit,
        merit_gradient,
        constraint_violation,
        solution_sensitivity,
    )
end
