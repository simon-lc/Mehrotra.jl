    function unpack_halfspaces(θ::Vector{T}, bundle_dimensions::Vector{Int}) where T
        nθ = 2 .+ 2 .* bundle_dimensions
        m = length(bundle_dimensions)

        A = [zeros(T, i, 2) for i in bundle_dimensions]
        b = [zeros(T, i) for i in bundle_dimensions]
        o = [zeros(T, i) for i in bundle_dimensions]

        off = 0
        for i = 1:m
            A[i], b[i], o[i] = unpack_halfspaces(θ[off .+ (1:nθ[i])])
            off += nθ[i]
        end
        return A, b, o
    end

    function pack_halfspaces(A::Vector{Matrix{T}}, b::Vector{Vector{T}}, o::Vector{Vector{T}}) where T
        m = length(b)
        bundle_dimensions = length.(b)
        nθ = 2 .+ 2 .* bundle_dimensions

        θ = zeros(T,sum(nθ))

        off = 0
        for i = 1:m
            θ[off .+ (1:nθ[i])] .= pack_halfspaces(A[i], b[i], o[i])
            off += nθ[i]
        end
        return θ, bundle_dimensions
    end

    function unpack_halfspaces(θ::Vector{T}) where T
        nθ = length(θ)
        n = Int(floor((nθ - 2)/2))

        A = zeros(T,n, 2)
        b = zeros(T,n)
        o = zeros(T,2)

        for i = 1:n
            A[i,:] .= [cos(θ[i]), sin(θ[i])]
        end
        b .= θ[n .+ (1:n)]
        o = θ[2n .+ (1:2)]
        return A, b, o
    end

    function pack_halfspaces(A::Matrix{T}, b::Vector{T}, o::Vector{T}) where T
        n = length(b)
        θ = zeros(T,2+2n)

        for i = 1:n
            θ[i] = atan(A[i,2], A[i,1])
        end
        θ[n .+ (1:n)] .= b
        θ[2n .+ (1:2)] .= o
        return θ
    end

    # bundle_dimensions0 = [1,2,3,4,22]
    # A0 = [rand(i,2) for i in bundle_dimensions0]
    # b0 = [rand(i) for i in bundle_dimensions0]
    # o0 = [rand(2) for i in bundle_dimensions0]
    # for i = 1:5
    #     for j = 1:bundle_dimensions0[i]
    #         A0[i][j,:] ./= norm(A0[i][j,:])
    #     end
    # end
    # θ1, bundle_dimensions1 = pack_halfspaces(A0, b0, o0)
    # A1, b1, o1 = unpack_halfspaces(θ1, bundle_dimensions1)
    # A1 .- A0
    # b1 .- b0
    # o1 .- o0

    function loss(WC, E, θ, bundle_dimensions; βb=1e-3, βo=1e-2, βf=1e-2, Δ=2e-2, δ=1e2)
        nb = length(bundle_dimensions)
        n = length(WC)
        m = size(WC[1], 2)
        N = n * m
        l = 0.0

        # halfspaces
        A, b, o = unpack_halfspaces(θ, bundle_dimensions)
        # add floor
        Af = [0 1.0]
        bf = [0.0]

        function ϕ(x)
            d = sdf(x, Af, bf, δ)
            for i = 1:nb
                # d = min(d, sdf(x, A[i], b[i] + A[i]*o[i], δ))
                d = min(d, sdf(x, A[i], b[i], δ))
            end
            return d
        end

        function ∇ϕ(x)
            imin = 0
            dmin = sdf(x, Af, bf, δ)
            for i = 1:nb
                d = sdf(x, A[i], b[i], δ)
                if d <= dmin
                    imin = i
                    dmin = d
                end
            end
            (imin == 0) && return gradient_sdf(x, Af, bf, δ)
            return sdf(x, A[imin], b[imin], δ)
        end

        # int, bnd, ext points
        for i = 1:n
            eyeposition = E[i]
            for j = 1:m
                xbnd = WC[i][2:3,j] # remove 3rd dimension which is always 0 in 2D
                v = xbnd - eyeposition[2:3]
                v ./= 1e-5 + norm(v)
                xint = xbnd + Δ * v
                xext = xbnd - Δ * v
                l += 0.5 * (ϕ(xbnd) - 0.0)^2 / n / m
                l += 0.5 * (ϕ(xint)/Δ + 1)^2 / n / m
                l += 0.5 * (ϕ(xext)/Δ - 1)^2 / n / m

                # l += 0.5 * norm(∇ϕ(x) + v)^2 / n / m
            end
        end

        # off = 0
        # for (i,ni) in enumerate(bundle_dimensions)
        #     θ[]
        # end
        # # floor interpenetration constraints, project point cloud onto the floor
        # # then enforce that those points are not in any polytope
        # for i = 1:n
        #     for j = 1:m
        #         x = [WC[i][2,j], 0]
        #         for k = 1:nb
        #             l += βf * max(0, -sdf(x, A[k], b[k] + A[k]*o[k], δ))^2 / n / m / nb
        #             l += βf * max(0, -sdf(x - [0, Δ], A[k], b[k] + A[k]*o[k], δ))^2 / n / m / nb
        #         end
        #     end
        # end

        # add regularization on b akin to cvxnet eq. 5
        l += βb * norm(b) / length(b)
        for i = 1:nb
            l += βo * norm(o[i])^2 / nb
        end
        return l
    end
