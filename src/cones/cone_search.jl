
function cone_search!(α, x, Δx, idx_nn, idx_soc; τ_nn=0.99, τ_soc=0.99, ϵ=1e-14, decoupling::Bool=false)
    # Non negative cone
    for (i,ii) in enumerate(idx_nn)
        if Δx[ii] < 0.0
            α[ii] = min(α[ii], - τ_nn * x[ii] / Δx[ii])
        end
    end
    # Second order cone
    # check Section 8.2 CVXOPT
    for idx in idx_soc
        second_order_cone_search!(α, x, Δx, idx; τ_soc=τ_soc, ϵ=ϵ)
    end
    !decoupling && (α .= minimum(α))
    return nothing
end

function second_order_cone_search!(α, x, Δx, idx; τ_soc=0.99, ϵ=1e-14)
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
            for i in idx
                α[i] = min(α[i], τ_soc / (ρv_norm - ρs))
            end
        end
    end
    return nothing
end


# num_cone = 15
# idx_nn = collect(1:3)
# idx_soc = [collect(4:6), collect(7:9), collect(10:12), collect(13:15)]
#
# x = rand(num_cone)
# Δx = rand(num_cone)
# α = 1.0
# α = ones(num_cone)
# cone_search(α, x, Δx, idx_nn, idx_soc)
# Main.@code_warntype cone_search(α, x, Δx, idx_nn, idx_soc)
# @benchmark $cone_search($α, $x, $Δx, $idx_nn, $idx_soc)
