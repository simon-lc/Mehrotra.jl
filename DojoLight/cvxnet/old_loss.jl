function mysolve!(θinit, loss, Gloss, Hloss, projection, clamping, nθ; max_iterations=60)
    θ = deepcopy(θinit)
    trace = [deepcopy(θ)]

    softness = 6.0
    reg = 1e+2
    ω_centroid = 1.0

    softness_min = 6.0
    softness_max = 100.0
    reg_min = 1e-3
    reg_max = 1e+2

    # newton's method
    for iterations = 1:max_iterations
        l = loss(θ, softness)
        core_loss = local_loss(θ, softness, ω_centroid=0.0, ω_offset=0.0)
        (core_loss < 1e-4) && break
        G = Gloss(θ, softness)
        H = Hloss(θ, softness)
        D = Diagonal(θdiag)
        @show norm(reg * D)

        Δθ = - (H + reg * D) \ G

        # linesearch
        α = 1.0
        for j = 1:10
            l_candidate = loss(projection(θ + clamping(α * Δθ)), softness)
            if l_candidate <= l
                softness = clamp(softness * 1.20, softness_min, softness_max)
                reg = clamp(reg/1.5, reg_min, reg_max)
                break
            end
            α /= 2
            if j == 10
                # softness = clamp(softness*1.05, softness_min, softness_max)
                # reg = clamp(reg/1.3, reg_min, reg_max)
                softness = clamp(softness/1.05, softness_min, softness_max)
                reg = clamp(reg*1.5, reg_min, reg_max)
                # α = α / 10

                # individual line search
                α = 1e-2*ones(np)
                off = 0
                for k = 1:length(nθ)
                    ind = off .+ (1:nθ[k]); off += nθ[k]
                    for ii = 1:20
                        θ_candidate = copy(θ)
                        θ_candidate[ind] .+= clamping(α[k] * Δθ)[ind]
                        l_candidate = loss(projection(θ_candidate), softness)
                        (l_candidate <= l) && break
                        α[k] /= 2
                    end
                end

            end
        end

        # header
        if rem(iterations - 1, 10) == 0
            @printf "-------------------------------------------------------------------\n"
            @printf "iter   core_loss   loss        step        reg         softness \n"
            @printf "-------------------------------------------------------------------\n"
        end
        # iteration information
        @printf("%3d   %9.2e   %9.2e   %9.2e   %9.2e   %9.2e \n",
            iterations,
            core_loss,
            l,
            mean(α),
            reg,
            softness)
        if typeof(α) <: Vector
            off = 0
            for k = 1:length(nθ)
                ind = off .+ (1:nθ[k]); off += nθ[k]
                θ[ind] += clamping(α[k] * Δθ)[ind]
            end
            θ = projection(θ)
        else
            θ = projection(θ + clamping(α * Δθ))
        end
        push!(trace, deepcopy(θ))
    end
    return θ, trace
end



# function loss(P, e, θ::Vector{T}, polytope_dimensions; βb=1e-3, βo=1e-2, βf=1e-2, βμ=1e0, βc=1e-2, Δ=2e-2, δ=1e2, top_k::Int=20) where T
#     nb = length(polytope_dimensions)
#     n = length(P)
#     m = size(P[1], 2)
#     N = n * m
#     l = 0.0
#
#     # halfspaces
#     A, b, o = unpack_halfspaces(θ, polytope_dimensions)
#     for i = 1:nb
#         l += βc * max(0, sdf(o[i], A[i], b[i], δ))
#     end
#     # add floor
#     Af = [0.0 1.0]
#     bf = [0.0]
#
#     function ϕ(x)
#         d = sdf(x, Af, bf, δ)
#         for i = 1:nb
#             d = min(d, sdf(x - o[i], A[i], b[i], δ))
#             # d = min(d, sdf(x, A[i], b[i], δ))
#         end
#         return d
#     end
#
#     function ∇ϕ(x)
#         imin = 0
#         dmin = sdf(x, Af, bf, δ)
#         for i = 1:nb
#             d = sdf(x - o[i], A[i], b[i], δ)
#             # d = sdf(x, A[i], b[i], δ)
#             if d <= dmin
#                 imin = i
#                 dmin = d
#             end
#         end
#         (imin == 0) && return gradient_sdf(x, Af, bf, δ)
#         return gradient_sdf(x - o[imin], A[imin], b[imin], δ)
#         # return gradient_sdf(x, A[imin], b[imin], δ)
#     end
#
#     indices = [zeros(Int, top_k) for i = 1:n]
#     distances = [zeros(T, m) for i=1:n]
#     for i = 1:n
#         distances[i] .= [0.5 * norm(o[i] - P[i][2:3,j])^2 for j = 1:m]
#         for j = 1:m
#             if P[i][3,j] < 1e-3
#                 distances[i][j] = +Inf
#             end
#         end
#         indices[i] .= partialsortperm(distances[i], 1:top_k)
#         for ii in indices[i]
#             d = distances[i][ii]
#             (d < Inf) && (l += 0.5 * d / top_k / nb * βμ)
#         end
#     end
#
#     # indices = [zeros(Int, m) for i = 1:n]
#     # for i = 1:n
#     #     distances = [0.5 * norm(o[i] - P[i][2:3,j])^2 for j = 1:m]
#     #     for j = 1:m
#     #         if P[i][3,j] < 1e-3
#     #             distance = +Inf
#     #         end
#     #     end
#     #     indices[i] .= partialsortperm(distances, 1:top_k)
#     #     for ii in indices[i]
#     #         l += 0.5 * distances[ii] / top_k / nb * βμ
#     #     end
#     # end
#
#     # int, bnd, ext points
#     for i = 1:n
#         eyeposition = e[i]
#         for j = 1:m
#             xbnd = P[i][2:3,j] # remove 3rd dimension which is always 0 in 2D
#             v = xbnd - eyeposition[2:3]
#             v ./= 1e-5 + norm(v)
#             xint = xbnd + Δ * v
#             xext = xbnd - Δ * v
#             l += 0.5 * (ϕ(xbnd) - 0.0*Δ)^2 / n / m
#             l += 0.5 * (ϕ(xint) + 1.0*Δ)^2 / n / m
#             l += 0.5 * (ϕ(xext) - 1.0*Δ)^2 / n / m
#         end
#     end
#
#     # off = 0
#     # for (i,ni) in enumerate(polytope_dimensions)
#     #     θ[]
#     # end
#     # # floor interpenetration constraints, project point cloud onto the floor
#     # # then enforce that those points are not in any polytope
#     # for i = 1:n
#     #     for j = 1:m
#     #         x = [P[i][2,j], 0]
#     #         for k = 1:nb
#     #             l += βf * max(0, -sdf(x, A[k], b[k] + A[k]*o[k], δ))^2 / n / m / nb
#     #             l += βf * max(0, -sdf(x - [0, Δ], A[k], b[k] + A[k]*o[k], δ))^2 / n / m / nb
#     #         end
#     #     end
#     # end
#
#     # add regularization on b akin to cvxnet eq. 5
#     l += βb * norm(b) / length(b)
#     for i = 1:nb
#         l += βo * norm(o[i])^2 / nb
#     end
#
#
#     nθ = 2 .+ 2 .* polytope_dimensions
#     off = 0
#     for i = 1:nb
#         mi = polytope_dimensions[i]
#         Aθ = θ[off .+ (1:mi)]
#         l += βf * 0.5 * norm(Aθ - range(-π, π, length=mi+1)[1:mi])^2 / nb
#         off += nθ[i]
#     end
#
#     return l
# end
