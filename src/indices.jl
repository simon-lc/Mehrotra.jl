struct Indices212
    variables::Vector{Int}
    cone::Vector{Int}
    cone_dual::Vector{Int}
    equality::Vector{Int}
    cone_product::Vector{Int}
    parameters::Vector{Int}
    total::Vector{Int}
    cone_nonnegative::Vector{Int}
    cone_second_order::Vector{Vector{Int}}
end

function Indices(num_variables, num_cone, num_parameters;
    nonnegative=collect(1:num_cone),
    second_order=[collect(1:0)])

    variables = collect(1:num_variables)
    cone = collect(num_variables + (1:num_cone))
    cone_dual = collect(num_variables + num_cone .+ (1:num_cone))

    equality = collect(1:num_variables + num_cone)
    cone_product = collect(num_variables + num_cone .+ (1:num_cone))

    parameters = collect(1:num_parameters)

    total = collect(1:num_variables + 2num_cone)

    return Indices212(
        variables,
        cone,
        cone_dual,
        equality,
        cone_product,
        parameters,
        total,
        nonnegative,
        second_order,
    )
end
