struct Dimensions
    variables::Int
    primals::Int
    duals::Int
    slacks::Int
    cone::Int
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

    num_duals = num_cone
    num_slacks = num_cone
    num_variables = num_primals + num_duals + num_slacks
    num_equality = num_primals + num_slacks

    Dimensions(
        num_variables,
        num_primals,
        num_duals,
        num_slacks,
        num_cone,
        num_equality,
        num_cone,
        num_parameters,
        nonnegative,
        second_order,
    )
end
