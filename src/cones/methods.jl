struct ConeMethods228{B,BX,P,PX,PXI,TA}
    barrier::B
    barrier_gradient::BX
    product::P
    product_jacobian::PX
    product_jacobian_inverse::PXI
    target::TA
end

function ConeMethods228(num_cone, idx_nn, idx_soc)
    Φ_func, Φa_func, p_func, pa_func, pai_func, t_func = generate_cones(num_cone, idx_nn, idx_soc)
    return ConeMethods228(
        Φ_func,
        Φa_func,
        p_func,
        pa_func,
        pai_func,
        t_func,
    )
end
