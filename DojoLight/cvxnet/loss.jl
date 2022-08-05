function unpack_halfspaces(θ::Vector, bundle_dimensions::Vector{Int})
    nθ = 2 .+ 2 .* bundle_dimensions
    m = length(bundle_dimensions)

    A = [zeros(i, 2) for i in bundle_dimensions]
    b = [zeros(i) for i in bundle_dimensions]
    o = [zeros(i) for i in bundle_dimensions]

    off = 0
    for i = 1:m
        A[i], b[i], o[i] = unpack_halfspaces(θ[off .+ (1:nθ[i])])
        off += nθ[i]
    end
    return A, b, o
end

function pack_halfspaces(A::Vector{<:Matrix}, b::Vector{<:Vector}, o::Vector{<:Vector})
    m = length(b)
    bundle_dimensions = length.(b)
    nθ = 2 .+ 2 .* bundle_dimensions

    θ = zeros(sum(nθ))

    off = 0
    for i = 1:m
        θ[off .+ (1:nθ[i])] .= pack_halfspaces(A[i], b[i], o[i])
        off += nθ[i]
    end
    return θ, bundle_dimensions
end

function unpack_halfspaces(θ::Vector)
    nθ = length(θ)
    n = Int(floor((nθ - 2)/2))

    A = zeros(n, 2)
    b = zeros(n)
    o = zeros(2)

    for i = 1:n
        A[i,:] .= [cos(θ[i]), sin(θ[i])]
    end
    b .= θ[n .+ (1:n)]
    o = θ[2n .+ (1:2)]
    return A, b, o
end

function pack_halfspaces(A::Matrix, b::Vector, o::Vector)
    n = length(b)
    θ = zeros(2+2n)

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

function loss(WC, E, bundle_dimensions, θ; βb=1e-3, βo=1e-2, Δ=2e-2, δ=1e2)
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

    # int, bnd, ext points
    # bound
    for i = 1:n
        eyeposition = E[i]
        for j = 1:m
            xbnd = WC[i][2:3,j] # remove 3rd dimension which is always 0 in 2D
            v = xbnd - eyeposition[2:3]
            v ./= 1e-5 + norm(v)
            xint = xbnd + Δ * v
            xext = xbnd - Δ * v
            l += 0.5 * (ϕ(xbnd) - 0.0)^2 / N
            l += 0.5 * (ϕ(xint)/Δ + 1)^2 / N
            l += 0.5 * (ϕ(xext)/Δ - 1)^2 / N
        end
    end

    # add regularization on b akin to cvxnet eq. 5
    l += βb * norm(b) / length(b)
    for i = 1:nb
        l += βo * norm(o[i]) / nb
    end
    return l
end
