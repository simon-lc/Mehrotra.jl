################################################################################
# signed distance function
################################################################################
function signed_distance(p, A, b, δ)
    N = length(b)
    v = δ[1] * (A*p - b)
    vm = maximum(v)
    e = exp.(v .- vm)
    s = 1/N * sum(e) # average
    ϕ = (log(s) + vm) / δ[1]
    return ϕ
end

function gradient_signed_distance(p, A, b, δ)
    N = length(b)
    v = δ[1] * (A*p - b)
    vm = maximum(v)
    e = exp.(v .- vm)
    s = 1/N * sum(e) # average
    ϕ = log(s) / δ[1]

    # g = 1/(δ * s) * ones(1,N)/N * Diagonal(e) * δ * A
    g = 1/(s * N) * A' * e
    return g
end

function hessian_signed_distance(p, A, b, δ)
    N = length(b)
    v = δ[1] * (A*p - b)
    vm = maximum(v)
    e = exp.(v .- vm)
    s = 1/N * sum(e) # average
    ϕ = log(s) / δ[1]


    dedx = δ * Diagonal(e) * A
    dsdx = 1/N * δ * e' * A
    H = 1/(s * N) * A' * dedx
    H += 1/N * A' * e * (-1/s^2) * dsdx
    return H
end

################################################################################
# squared signed distance function
################################################################################
function squared_signed_distance(p, A, b, δ)
    0.5 * signed_distance(p, A, b, δ)^2
end

function gradient_squared_signed_distance(p, A, b, δ)
    signed_distance(p, A, b, δ) * gradient_signed_distance(p, A, b, δ)
end

function hessian_squared_signed_distance(p, A, b, δ)
    ϕ = signed_distance(p, A, b, δ)
    g = gradient_signed_distance(p, A, b, δ)
    H = hessian_signed_distance(p, A, b, δ)
    return ϕ * H + g * g'
end

function signed_distance(p::Vector{T}, poly::Polyhedron{T,N,D}) where {T,N,D}
    A = poly.A
    b = poly.b
    δ = poly.δ
    signed_distance(p, A, b, δ)
end

function squared_signed_distance(p::Vector{T}, poly::Polyhedron{T,N,D}) where {T,N,D}
    A = poly.A
    b = poly.b
    δ = poly.δ
    squared_signed_distance(p, A, b, δ)
end
