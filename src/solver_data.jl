
struct SolverData212{T}
    residual::Vector{T}
    jacobian_variables::SparseMatrixCSC{T,Int}
    jacobian_parameters::Matrix{T}
    step::Vector{T}
    step_correction::Vector{T}
    merit::Vector{T}
    merit_gradient::Vector{T}
    constraint_violation::Vector{T}
    solution_sensitivity::Matrix{T}
end

function SolverData(dims::Dimensions212, idx::Indices212;
    T=Float64)

    num_variables = dims.variables
    num_parameters = dims.parameters
    num_equality = dims.equality_dual
    num_cone = dims.cone_dual

    num_total = dims.total
    num_symmetric = dims.symmetric

    residual = Point(dims, idx)

    jacobian_variables = spzeros(num_total, num_total)
    jacobian_parameters = zeros(num_total, num_parameters)

    step = Point(dims, idx)
    step_correction = Point(dims, idx)

    merit = zeros(1)
    merit_gradient = zeros(num_variables + num_equality + num_cone)

    constraint_violation = zeros(num_equality + num_cone)

    solution_sensitivity = zeros(num_total, num_parameters)

    SolverData212(
        residual,
        jacobian_variables,
        jacobian_parameters,
        step,
        step_correction,
        merit,
        merit_gradient,
        constraint_violation,
        solution_sensitivity,
    )
end
