function trans_intersection(e::SVector{Ne,T}, v::SVector{Nv,T}, ρ::T, A::Matrix{}, b::Vector, o::Vector) where {Ne,Nv,T,Td}
	n = length(b)
    eoff = e - o
	αmin = +Inf
    αmax = -Inf

	c = 0
    for i = 1:n
        denum = (A[i,:]' * v)
        (abs(denum) < 1e-5) && continue
        α = (b[i] - A[i,:]' * eoff) / denum
        x = eoff + α .* v
        s = maximum(A * x .- b)
		if s <= 1e-10
			c += 1
			αmin = min(αmin, α)
			αmax = max(αmax, α)
		end
    end
	(c == 1) && (αmax = +Inf)
    return [αmin, αmax]
end

function trans_intersection(e::Vector, v::Vector, ρ, A::Vector{<:Matrix{T}},
        b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}}) where T

    np = length(b)
	α = zeros(T,2,np)
	off = 0
    for i = 1:np
		α[:,i] = trans_intersection(e, v, ρ, A[i], b[i], o[i])
    end
	return α
end

function trans_point_cloud(e::Vector, β, ρ, A::Vector{<:Matrix{T}},
        b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}}) where T

	nβ = length(β)
	d = zeros(2,nβ)

	off = 0
	for i = 1:nβ
		v = [cos(β[i]), sin(β[i])]
		α = trans_intersection(e, v, ρ, A, b, o)
		d[:,i] = trans_point_cloud(e, v, ρ, α)
	end
	return d
end

function trans_point_cloud(e::Vector, v::Vector, ρ, α::Matrix{T}) where T
	np = size(α, 2)
	α = α[:, sortperm(α[1, :])]

	α_trans = 0.0
	cum_e = 1.0
	for i = 1:np
		αmin = α[1,i]
		αmax = α[2,i]
		(αmin <= 0) && continue
		(αmin == Inf) && continue
		δ = αmax - αmin
		ex = exp(-δ*ρ)
		α_trans += αmin * (1 - ex) * cum_e
		cum_e *= ex
    end
	d = e + α_trans .* v
	return d
end

function trans_point_loss(e::Vector, v::Vector, ρ, α::Matrix{T}, d̂::Vector) where T
	d = trans_point_cloud(e, v, ρ, α)
	return 0.5 * (d - d̂)' * (d - d̂) + 0.5 * softabs(norm(d - d̂), δ=0.001)
end

function trans_point_loss(e::Vector, v::Vector, ρ, θ::Vector{T},
		polytope_dimensions::Vector{Int}, d̂::Vector) where T

	np = length(polytope_dimensions)
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)

 	α = trans_intersection(e, v, ρ, A, b, o)
	l = trans_point_loss(e, v, ρ, α, d̂)
	return l
end

function trans_point_loss(e::Vector{<:Vector}, β::Vector, ρ, θ::Vector,
		polytope_dimensions::Vector{Int}, d::Vector{<:Matrix})

    ne = length(e)
    l = 0.0
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    for i = 1:ne
		nβ = length(β[i])
		for j = 1:nβ
			v = [cos(β[i][j]), sin(β[i][j])]
			lj = trans_point_loss(e[i], v, ρ, θ, polytope_dimensions, d[i][:,j])
			l += lj / nβ / ne
		end
    end
    return l
end

function point_cloud_smoothing(vis::Visualizer, e, β, A, b, o)
	set_floor!(vis)
	set_background!(vis)
	set_light!(vis)

	anim = MeshCat.Animation(20)
	for (i, ρ) in enumerate(range(0, 3.0, length=100))
		atframe(anim, i) do
			d = trans_point_cloud(e, β, exp(log(10)*ρ), A, b, o)
			num_points = size(d, 2)
			set_2d_point_cloud!(vis, [e], [d]; name=:point_cloud)
		end
	end
	setanimation!(vis, anim)
	return vis, anim
end
