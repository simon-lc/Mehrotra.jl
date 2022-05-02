struct ProblemMethods212{T,E,EX,EP,C,CX,CP,CD,CDX,CDXX,CDXP}
    equality_constraint::E                             # g
    equality_jacobian_variables::EX                    # gx
    equality_jacobian_parameters::EP                   # gθ
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
    cone_constraint::C                                 # h
    cone_jacobian_variables::CX                        # hx
    cone_jacobian_parameters::CP                       # hθ
    cone_jacobian_variables_cache::Vector{T}
    cone_jacobian_parameters_cache::Vector{T}
    cone_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    cone_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
    cone_product::CD                                      # h'z
    cone_product_jacobian_variables::CDX                  # (h'y)x
    cone_product_jacobian_variables_variables::CDXX       # (h'y)xx
    cone_product_jacobian_variables_parameters::CDXP      # (h'y)xθ
    cone_product_jacobian_variables_variables_cache::Vector{T}       # (h'y)xx
    cone_product_jacobian_variables_parameters_cache::Vector{T}      # (h'y)xθ
    cone_product_jacobian_variables_variables_sparsity::Vector{Tuple{Int,Int}}       # (h'y)xx
    cone_product_jacobian_variables_parameters_sparsity::Vector{Tuple{Int,Int}}      # (h'y)xθ
end

function ProblemMethods(num_variables::Int, num_parameters::Int, equality::Function, cone::Function)
    g, gx, gθ, gx_sparsity, gθ_sparsity, gᵀy, gᵀyx, gᵀyxx, gᵀyxθ, gᵀyxx_sparsity, gᵀyxθ_sparsity, num_g = generate_gradients(equality, num_variables, num_parameters, :vector)
    h, hx, hθ, hx_sparsity, hθ_sparsity, hᵀy, hᵀyx, hᵀyxx, hᵀyxθ, hᵀyxx_sparsity, hᵀyxθ_sparsity, num_h = generate_gradients(cone, num_variables, num_parameters, :vector)

    methods = ProblemMethods212(
        g, gx, gθ,
            zeros(length(gx_sparsity)), zeros(length(gθ_sparsity)),
            gx_sparsity, gθ_sparsity,
        h, hx, hθ,
            zeros(length(hx_sparsity)), zeros(length(hθ_sparsity)),
            hx_sparsity, hθ_sparsity,
        hᵀy, hᵀyx, hᵀyxx, hᵀyxθ,
            zeros(length(hᵀyxx_sparsity)), zeros(length(hᵀyxθ_sparsity)),
            hᵀyxx_sparsity, hᵀyxθ_sparsity,
    )

    return methods, num_g, num_h
end
