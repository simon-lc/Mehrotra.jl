struct Dimensions212
    variables::Int
    cone::Int
    equality::Int
    cone_product::Int
    parameters::Int
    total::Int
    cone_nonnegative::Int
    cone_second_order::Vector{Int}
end

function Dimensions(num_variables, num_cone, num_parameters;
    nonnegative=num_cone,
    second_order=[0,],
    )

    num_total = num_variables + num_cone # primals
    num_total += num_cone                # duals

    Dimensions212(
        num_variables,
        num_cone,
        num_variables + num_cone,
        num_cone,
        num_parameters,
        num_total,
        nonnegative,
        second_order,
    )
end
