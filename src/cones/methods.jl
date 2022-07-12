struct ConeMethods{T,B,BX,P,PX,PXI,TA}
    barrier::B
    barrier_gradient::BX
    product::P
    product_jacobian::PX
    product_jacobian_inverse::PXI
    target::TA
    product_jacobian_duals_cache::Vector{T}
    product_jacobian_slacks_cache::Vector{T}
    product_jacobian_inverse_duals_cache::Vector{T}
    product_jacobian_inverse_slacks_cache::Vector{T}
    product_jacobian_duals_sparsity::Vector{Tuple{Int,Int}}
    product_jacobian_slacks_sparsity::Vector{Tuple{Int,Int}}
    product_jacobian_inverse_duals_sparsity::Vector{Tuple{Int,Int}}
    product_jacobian_inverse_slacks_sparsity::Vector{Tuple{Int,Int}}
end

function ConeMethods(num_cone, idx_nn, idx_soc)
    Φ_func, Φa_func, p_func, pa_func, pai_func, t_func, pa_sparsity, pai_sparsity = generate_cones(num_cone, idx_nn, idx_soc)
    return ConeMethods(
        Φ_func,
        Φa_func,
        p_func,
        pa_func,
        pa_sparse_func,
        pai_func,
        t_func,
        zeros(length(pa_sparsity)),
        zeros(length(pa_sparsity)),
        zeros(length(pai_sparsity)),
        zeros(length(pai_sparsity)),
        pa_sparsity,
        pa_sparsity,
        pai_sparsity,
        pai_sparsity,
    )
end
