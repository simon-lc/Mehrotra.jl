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
        if length(idx) > 0
            x0 = x[idx[1]]
            x1 = x[idx[2:end]]
            Δx0 = Δx[idx[1]]
            Δx1 = Δx[idx[2:end]]

            x_x = max(x0^2 - x1' * x1, ϵ) + ϵ
            x_Δ = x0 * Δx0 - x1' * Δx1 + ϵ

            ρs = x_Δ / x_x
            ρv = Δx1 / sqrt(x_x)
            ρv -= (x_Δ / sqrt(x_x) + Δ0) / (x0 / sqrt(x_x) + 1) * x1 / x_x
            if norm(ρv) - ρs > 0.0
                α = min(α, τ_soc / (norm(ρv) - ρs))
            end
        end
    end
    return α
end
