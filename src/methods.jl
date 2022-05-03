struct ProblemMethods218{T,E,EX,EP}
    equality_constraint::E                             # g
    equality_jacobian_variables::EX                    # gx
    equality_jacobian_parameters::EP                   # gθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
    # cone_product::C                                    # h
    # cone_jacobian_variables::CX                        # hx
    # cone_jacobian_parameters::CP                       # hθ
    # cone_jacobian_variables_cache::Vector{T}
    # cone_jacobian_parameters_cache::Vector{T}
    # cone_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    # cone_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
    # cone_product::CD                                      # h'z
    # cone_product_jacobian_variables::CDX                  # (h'y)x
    # cone_product_jacobian_variables_variables::CDXX       # (h'y)xx
    # cone_product_jacobian_variables_parameters::CDXP      # (h'y)xθ
    # cone_product_jacobian_variables_variables_cache::Vector{T}       # (h'y)xx
    # cone_product_jacobian_variables_parameters_cache::Vector{T}      # (h'y)xθ
    # cone_product_jacobian_variables_variables_sparsity::Vector{Tuple{Int,Int}}       # (h'y)xx
    # cone_product_jacobian_variables_parameters_sparsity::Vector{Tuple{Int,Int}}      # (h'y)xθ
end

function ProblemMethods(equality::Function, dim::Dimensions218, ind::Indices218)
    f, fx, fθ, fx_sparsity, fθ_sparsity = generate_gradients(equality, dim, ind)

    methods = ProblemMethods218(
        f, fx, fθ,
            zeros(length(fx_sparsity)), zeros(length(fθ_sparsity)),
            fx_sparsity, fθ_sparsity,
        # h, hx, hθ,
        #     zeros(length(hx_sparsity)), zeros(length(hθ_sparsity)),
        #     hx_sparsity, hθ_sparsity,
        # hᵀy, hᵀyx, hᵀyxx, hᵀyxθ,
        #     zeros(length(hᵀyxx_sparsity)), zeros(length(hᵀyxθ_sparsity)),
        #     hᵀyxx_sparsity, hᵀyxθ_sparsity,
    )

    return methods
end
