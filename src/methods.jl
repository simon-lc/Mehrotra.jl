struct ProblemMethods228{T,E,EX,EP}
    equality_constraint::E                             # e
    equality_jacobian_variables::EX                    # ex
    equality_jacobian_parameters::EP                   # eθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
    # cone_product::C                                    # c
    # cone_jacobian_variables::CX                        # cx
    # cone_jacobian_parameters::CP                       # cθ
    # cone_jacobian_variables_cache::Vector{T}
    # cone_jacobian_parameters_cache::Vector{T}
    # cone_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    # cone_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function ProblemMethods(equality::Function, dim::Dimensions228, idx::Indices228)
    e, ex, eθ, ex_sparsity, eθ_sparsity = generate_gradients(equality, dim, idx)
    # c, cx, cθ, cx_sparsity, cθ_sparsity = generate_gradients(cone, dim, idx)

    methods = ProblemMethods228(
        e,
        ex,
        eθ,
        zeros(length(ex_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        eθ_sparsity,
        # c, cx, cθ,
        #     zeros(length(cx_sparsity)), zeros(length(cθ_sparsity)),
        #     cx_sparsity, cθ_sparsity,
    )

    return methods
end
