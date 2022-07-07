struct Indices
    variables::Vector{Int}
    primals::Vector{Int}
    duals::Vector{Int}
    slacks::Vector{Int}
    equality::Vector{Int}
    cone_product::Vector{Int}
    parameters::Vector{Int}
    cone_nonnegative::Vector{Int}
    cone_second_order::Vector{Vector{Int}}
end


function Indices(num_primals, num_cone, num_parameters;
    nonnegative=collect(1:num_cone),
    second_order=[collect(1:0)])

    num_variables = num_primals + 2 * num_cone
    variables = collect(1:num_variables)
    primals = collect(1:num_primals)
    duals = collect(num_primals .+ (1:num_cone))
    slacks = collect(num_primals + num_cone .+ (1:num_cone))

    equality = collect(1:num_primals + num_cone)
    cone_product = collect(num_primals + num_cone .+ (1:num_cone))

    parameters = collect(1:num_parameters)

    return Indices(
        variables,
        primals,
        duals,
        slacks,
        equality,
        cone_product,
        parameters,
        nonnegative,
        second_order,
    )
end
