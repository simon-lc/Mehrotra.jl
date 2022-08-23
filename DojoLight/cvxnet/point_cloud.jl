function julia_intersection(e::Vector, v::Vector, δ, A::Matrix, b::Vector, o::Vector)
    n = length(b)
    eoff = e - o
    αmin = +Inf

    for i = 1:n
        denum = (A[i,:]' * v)
        (abs(denum) < 1e-3) && continue
        α = (b[i] - A[i,:]' * eoff) / denum
        x = eoff + α .* v
        s = maximum(A * x .- b)
        (s <= 1e-10) && (αmin = min(αmin, α))
    end

	# add smoothness correction
	if αmin < +Inf && length(b) > 1
		x = eoff + αmin * v
		c = A * x - b
		cmax, imax = findmax(c)
		c = c[setdiff(1:length(c), imax)]
		cmax2, imax2 = findmax(c)
		αmin += 1/δ * exp(20(cmax2 - cmax))
	end

	#TODO instead of scaling this correction by 1/δ maybe we should use the thickness of the object along the ray.
    return αmin
end

function julia_intersection(e::Vector, v::Vector, δ, A::Vector{<:Matrix{T}},
        b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}}) where T

    np = length(b)
    α = zeros(T,np)
    for i = 1:np
        n = length(b[i])
        α[i] = julia_intersection(e, v, δ, A[i], b[i], o[i])
    end
    αsoft = -softmax(-α, δ)
    return αsoft
end

function julia_intersection(e::Vector, β::Number, δ, A::Vector{<:Matrix},
		b::Vector{<:Vector}, o::Vector{<:Vector})
    v = [cos(β), sin(β)]
    α = julia_intersection(e, v, δ, A, b, o)
    d = e + α * v
    return d
end

function julia_point_cloud(e::Vector, β, δ, A::Vector{Matrix{T}},
		b::Vector{<:Vector}, o::Vector{<:Vector}) where T
    nβ = length(β)
    d = zeros(T, 2, nβ)
    julia_point_cloud!(d, e, β, δ, A, b, o)
    return d
end

function julia_point_cloud!(d::Matrix, e::Vector, β, δ, A::Vector{<:Matrix},
		b::Vector{<:Vector}, o::Vector{<:Vector})
	nβ = length(β)
    for i = 1:nβ
        v = [cos(β[i]), sin(β[i])]
        α = julia_intersection(e, v, δ, A, b, o)
        d[:,i] = e + α * v
    end
    return nothing
end

function julia_loss(d::Vector{<:Matrix}, e::Vector{<:Vector}, β::Vector, δ, θ::Vector{T},
        polytope_dimensions::Vector{Int}) where T

    ne = length(e)
    l = 0.0
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    for i = 1:ne
        dθ = julia_point_cloud(e[i], β[i], δ, A, b, o)
        l += 0.5 * norm(d[i] - dθ)^2 / size(dθ, 2)
    end
    return l
end
