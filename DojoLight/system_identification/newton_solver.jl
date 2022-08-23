function newton_solver!(θinit, loss, Gloss, Hloss, projection, clamping;
        max_iterations=20,
        reg_min=1e-4,
        reg_max=1e+0,
        reg_step=2.0,
        residual_tolerance=1e-4,
        D=Diagonal(ones(length(θinit))))

    θ = deepcopy(θinit)
    trace = [deepcopy(θ)]
    reg = reg_max

    # newton's method
    for iterations = 1:max_iterations
        l = loss(θ)
        (l < residual_tolerance) && break
        G = Gloss(θ)
        H = Hloss(θ)

        # reg = clamp(norm(G, Inf)/10, reg_min, reg_max)
        Δθ = - (H + reg * D) \ G

        # linesearch
        α = 1.0
        for j = 1:10
            l_candidate = loss(projection(θ + clamping(α * Δθ)))
            if l_candidate <= l
                reg = clamp(reg/reg_step, reg_min, reg_max)
                break
            end
            α /= 2
            if j == 10
                reg = clamp(reg*reg_step, reg_min, reg_max)
            end
        end

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
            norm(clamping(α * Δθ), Inf),
            norm(G, Inf),
            reg,
            )
        θ = projection(θ + clamping(α * Δθ))
        push!(trace, deepcopy(θ))
    end
    return θ, trace
end
