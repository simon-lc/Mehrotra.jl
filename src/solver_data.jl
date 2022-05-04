struct SolverData228{T}
    residual::Point228{T}
    jacobian_variables::SparseMatrixCSC{T,Int}
    jacobian_parameters::Matrix{T}
    step::Point228{T}
    step_correction::Point228{T}
    merit::Vector{T}
    merit_gradient::Vector{T}
    constraint_violation::Vector{T}
    solution_sensitivity::Matrix{T}
end

function SolverData(dim::Dimensions228, idx::Indices228;
    T=Float64)

    num_variables = dim.variables
    num_parameters = dim.parameters
    num_equality = dim.equality
    num_cone = dim.cone

    residual = Point(dim, idx)

    jacobian_variables = spzeros(num_variables, num_variables)
    jacobian_parameters = zeros(num_variables, num_parameters)

    step = Point(dim, idx)
    step_correction = Point(dim, idx)

    merit = zeros(1)
    merit_gradient = zeros(num_variables)

    constraint_violation = zeros(num_variables)

    solution_sensitivity = zeros(num_variables, num_parameters)

    SolverData228(
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
