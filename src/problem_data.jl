struct ProblemData212{T,X}
    equality_constraint::Vector{T}
    equality_jacobian_variables::Matrix{T}
    equality_jacobian_parameters::Matrix{T}
    cone_constraint::Vector{T}
    cone_jacobian_variables::Matrix{T}
    cone_jacobian_parameters::Matrix{T}
    cone_product::Vector{T}
    cone_product_jacobian_primal::Matrix{T}
    cone_product_jacobian_dual::Matrix{T}
    custom::X
end

function ProblemData(num_variables, num_parameters, num_equality, num_cone;
    custom=nothing)

    equality_constraint = zeros(num_equality)
    equality_jacobian_variables = zeros(num_equality, num_variables)
    equality_jacobian_parameters = zeros(num_equality, num_parameters)

    cone_constraint = zeros(num_cone)
    cone_jacobian_variables = zeros(num_cone, num_variables)
    cone_jacobian_parameters = zeros(num_cone, num_parameters)

    cone_product = zeros(num_cone)
    cone_product_jacobian_primal = zeros(num_cone, num_cone)
    cone_product_jacobian_dual = zeros(num_cone, num_cone)

    ProblemData212(
        equality_constraint,
        equality_jacobian_variables,
        equality_jacobian_parameters,
        cone_constraint,
        cone_jacobian_variables,
        cone_jacobian_parameters,
        cone_product,
        cone_product_jacobian_primal,
        cone_product_jacobian_dual,
        custom,
    )
end
