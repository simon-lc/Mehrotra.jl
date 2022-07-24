struct ConeMethods{T,B,BX,P,PX}
    barrier::B
    barrier_gradient::BX
    product::P
    product_jacobian::PX
    product_jacobian_duals_cache::Vector{T}
    product_jacobian_slacks_cache::Vector{T}
    product_jacobian_sparsity::Vector{Tuple{Int,Int}}
end

function ConeMethods(num_cone, idx_nn, idx_soc)
    Φ_func, Φa_func, p_func, pa_func, pa_sparsity = generate_cones(num_cone, idx_nn, idx_soc)
    return ConeMethods(
        Φ_func,
        Φa_func,
        p_func,
        pa_func,
        zeros(length(pa_sparsity)),
        zeros(length(pa_sparsity)),
        pa_sparsity,
    )
end
