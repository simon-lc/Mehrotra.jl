struct ProblemData{T,X}
    equality_constraint::Vector{T}
    equality_jacobian_variables::Matrix{T}
    equality_jacobian_parameters::Matrix{T}
    cone_product::Vector{T}
    cone_product_jacobian_dual::Matrix{T}
    cone_product_jacobian_slack::Matrix{T}
    cone_product_jacobian_inverse_dual::Matrix{T}
    cone_product_jacobian_inverse_slack::Matrix{T}
    cone_target::Vector{T}
    # compressed_jacobian_variables::Matrix{T}
    # cone_jacobian_variables::Matrix{T}
    # cone_jacobian_parameters::Matrix{T}
    custom::X
end

function ProblemData(num_variables, num_parameters, num_equality, num_cone;
    custom=nothing)

    equality_constraint = zeros(num_equality)
    equality_jacobian_variables = zeros(num_equality, num_variables)
    equality_jacobian_parameters = zeros(num_equality, num_parameters)

    cone_product = zeros(num_cone)
    cone_product_jacobian_dual = zeros(num_cone, num_cone)
    cone_product_jacobian_slack = zeros(num_cone, num_cone)
    cone_product_jacobian_inverse_dual = zeros(num_cone, num_cone)
    cone_product_jacobian_inverse_slack = zeros(num_cone, num_cone)
    cone_target = zeros(num_cone)
    # compressed_jacobian_variables = zeros(num_equality, num_equality)
    # cone_jacobian_variables = zeros(num_cone, num_variables)
    # cone_jacobian_parameters = zeros(num_cone, num_parameters)

    ProblemData(
        equality_constraint,
        equality_jacobian_variables,
        equality_jacobian_parameters,
        cone_product,
        cone_product_jacobian_dual,
        cone_product_jacobian_slack,
        cone_product_jacobian_inverse_dual,
        cone_product_jacobian_inverse_slack,
        cone_target,
        # compressed_jacobian_variables,
        # cone_jacobian_variables,
        # cone_jacobian_parameters,
        custom,
    )
end
