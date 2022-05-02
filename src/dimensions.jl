struct Dimensions214
    variables::Int
    primals::Int
    cone::Int
    slack::Int
    equality::Int
    cone_product::Int
    parameters::Int
    cone_nonnegative::Int
    cone_second_order::Vector{Int}
end

function Dimensions(num_primals, num_cone, num_parameters;
    nonnegative=num_cone,
    second_order=[0,],
    )

    num_slack = num_cone
    num_variables = num_primals + num_cone + num_slack
    num_equality = num_primals + num_slack

    Dimensions214(
        num_variables,
        num_primals,
        num_cone,
        num_slack,
        num_equality,
        num_cone,
        num_parameters,
        nonnegative,
        second_order,
    )
end
