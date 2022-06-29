
function cone_search(α, x, Δx, idx_nn, idx_soc; τ_nn=0.99, τ_soc=0.99, ϵ=1e-14)
    # Non negative cone
    for (i,ii) in enumerate(idx_nn)
        if Δx[ii] < 0.0
            α = min(α, - τ_nn * x[ii] / Δx[ii])
        end
    end
    # Second order cone
    # check Section 8.2 CVXOPT
    for idx in idx_soc
        α = second_order_cone_search(x, Δx, idx, α; τ_soc=τ_soc, ϵ=ϵ)
    end
    return α
end

function second_order_cone_search(x, Δx, idx, α; τ_soc=0.99, ϵ=1e-14)
    # check Section 8.2 CVXOPT
    if length(idx) > 0
        x0 = x[idx[1]]
        Δx0 = Δx[idx[1]]

        x_x = 0.0
        x_Δ = 0.0
        for (i, ii) in enumerate(idx)
            x_x += (i==1) ? x[ii]*x[ii] : -x[ii]*x[ii]
            x_Δ += (i==1) ? x[ii]*Δx[ii] : -x[ii]*Δx[ii]
        end
        x_x = max(x_x, ϵ) + ϵ
        x_Δ = x_Δ + ϵ


        ρs = x_Δ / x_x
        ρv_norm = 0.0
        for (i, ii) in enumerate(idx)
            (i == 1) && continue
            ρvi = Δx[ii] / sqrt(x_x)
            ρvi -= (x_Δ / sqrt(x_x) + Δx0) / (x0 / sqrt(x_x) + 1) * x[ii] / x_x
            ρv_norm += ρvi^2
        end
        ρv_norm = sqrt(ρv_norm)

        if ρv_norm - ρs > 0.0
            α = min(α, τ_soc / (ρv_norm - ρs))
        end
    end
    return α
end
