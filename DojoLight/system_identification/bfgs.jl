function bfgs_solver!(xinit, loss, grad, projection, step_projection;
        max_iterations=20,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=2.0,
        line_search_iterations=15,
        line_search_schedule=0.5,
        loss_tolerance=1e-4,
        grad_tolerance=1e-4,
        H=Diagonal(ones(length(xinit))))

    n = length(xinit)
    # initialization
    stall = 0
    x = deepcopy(xinit)
    g_previous = zeros(n)
    trace = [deepcopy(x)]
    reg = reg_max

    # BFGS
    # https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf
    # Algorithm 6.1
    # Procedure 18.2
    for iterations = 1:max_iterations
        (stall >= 5) && break
        l = loss(x)
        (l < loss_tolerance) && break
        g = grad(x)
        (norm(g, Inf) < grad_tolerance) && break

        # Δx = - (H + reg * D) \ g
        Δx = - H * g

        # linesearch
        α = 1.0
        for j = 1:line_search_iterations
            l_candidate = loss(projection(x + step_projection(α * Δx)))
            if l_candidate <= l
                reg = clamp(reg/reg_step, reg_min, reg_max)
                stall = 0
                break
            end
            α *= line_search_schedule
            if j == 10
                stall += 1
                # α = 0.0
                reg = clamp(reg*exp(3.0*log(reg_step)), reg_min, reg_max)
            end
        end

        s = step_projection(α * Δx)
        y = g - g_previous

        ρ = 1 / (y' * s)
        if 1 / ρ > 0
            H .= (I - ρ * s * y') * H * (I - ρ * y * s') + ρ * s * s'
        else
            nothing
        end
        # Bs = (H \ s)
        # if s'* y >= 0.2 * s' *
        #     θ = 1.0
        # else
        #     θ = (0.8 * s' * Bs) / (s' * Bs - s' * y)
        # end
        # r = θ * y + (1 - θ) * Bs
        # ρr = 1 / (r' * s)
        # H .= (I - ρ * s * r') * H * (I - ρ * r * s') + ρ * s * s'


        # header
        if rem(iterations - 1, 10) == 0
            @printf "-------------------------------------------------------------------\n"
            @printf "iter   loss        step        |step|∞     |grad|∞     reg         \n"
            @printf "-------------------------------------------------------------------\n"
        end
        # iteration information
        @printf("%3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e\n",
            iterations,
            l,
            mean(α),
            norm(step_projection(α * Δx), Inf),
            norm(g, Inf),
            reg,
            )
        x = projection(x + step_projection(α * Δx))
        @show norm(step_projection(α * Δx))
        push!(trace, deepcopy(x))
        g_previous .= g
    end
    return x, trace
end
