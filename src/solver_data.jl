struct SolverData214{T}
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

function SolverData(dims::Dimensions214, idx::Indices214;
    T=Float64)

    num_variables = dims.variables
    num_parameters = dims.parameters
    num_equality = dims.equality
    num_cone = dims.cone

    residual = zeros(num_variables)

    jacobian_variables = spzeros(num_variables, num_variables)
    jacobian_parameters = zeros(num_variables, num_parameters)

    step = zeros(num_variables)
    step_correction = zeros(num_variables)

    merit = zeros(1)
    merit_gradient = zeros(num_variables)

    constraint_violation = zeros(num_variables)

    solution_sensitivity = zeros(num_variables, num_parameters)

    SolverData214(
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
