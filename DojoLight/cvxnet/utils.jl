################################################################################
# initialization
################################################################################
function parameter_initialization(d, polytope_dimensions; altitude_threshold=0.1)
	np = length(polytope_dimensions)
	# point cloud reaching the object
	d_object = []
	for i = 1:size(d,2)
	    di = d[:,i]
	    if di[2] > altitude_threshold
	        push!(d_object, di)
	    end
	end
	d_object = hcat(d_object...)
	# k-mean clustering
	kmres = kmeans(d_object, np)
	# initialization
	b_mean = 2 * mean(sqrt.(kmres.costs))
	θ = zeros(0)
	for i = 1:np
	    angles = range(-π, π, length=nh+1)[1:end-1]
		θi = [vcat([[cos(a), sin(a)] for a in angles]...); b_mean*ones(nh); kmres.centers[:, i]]
	    # θi = [vcat([[a] for a in angles]...); b_mean*ones(nh); kmres.centers[:, i]]
	    A, b, o = unpack_halfspaces(θi)
	    push!(θ, pack_halfspaces(A, b, o)...)
	end

	return θ, d_object, kmres
end


################################################################################
# projection
################################################################################
function projection(θ, polytope_dimensions)
	θmin = []
	θmax = []
    A, b, o = unpack_halfspaces(θ, polytope_dimensions)

    for (i,nh) in enumerate(polytope_dimensions)
		push!(θmin, [-1.0 * ones(2nh); +0.05 * ones(nh); -3.0 * ones(2)]...)
		push!(θmax, [+1.0 * ones(2nh); +0.40 * ones(nh); +3.0 * ones(2)]...)
        for j = 1:nh
            A[i][j,:] = A[i][j,:] / (1e-6 + norm(A[i][j,:]))
        end
    end

    return clamp.(θ, θmin, θmax)
end


################################################################################
# clamping
################################################################################
function clamping(Δθ, polytope_dimensions)
	Δθmin = []
	Δθmax = []
	for nh in polytope_dimensions
		push!(Δθmin, [-0.60 * ones(2nh); -0.05 * ones(nh); -0.05 * ones(2)]...)
		push!(Δθmax, [+0.60 * ones(2nh); +0.05 * ones(nh); +0.05 * ones(2)]...)
	end
    return clamp.(Δθ, Δθmin, Δθmax)
end

################################################################################
# loss
################################################################################
function shape_loss(θ, e, β, ρ, d_ref;
	δ=100.0,
	altitude_threshold=0.1,
	rendering=10.0,
	sdf_matching=1.0,
	overlap=0.1,
	individual=1.0,
	expansion=0.3,
	side_regularization=2.0,
	)

	θ_f, polytope_dimensions_f = add_floor(θ, polytope_dimensions)
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
	A_f, b_f, o_f = unpack_halfspaces(θ_f, polytope_dimensions_f)

	l = rendering * trans_point_loss([e], [β], ρ, θ_f, polytope_dimensions_f, [d_ref])

	# regularization
	l += side_regularization * 10.0 * sum([0.5*norm(bi .- 0.50)^2 for bi in b]) / (np * nh)

	# sdf matching
	for i = 1:nβ
		p = d_ref[:,i]
		ϕ = sdf(p, A_f, b_f, o_f, δ)
		l += sdf_matching * 0.1 * (ϕ^2 + abs(ϕ)) / nβ
	end

	# individual
	for i = 1:nβ
		di = d_ref[:,i]
		if di[2] > altitude_threshold
			idx = findmin([norm(oi - di) for oi in o])[2]
			ϕ = sdf(di, A[idx], b[idx], o[idx], δ)
			l += individual * 10.0 * (ϕ^2 + abs(ϕ)) / nβ
		end
	end

	# inside sampling, overlap penalty
	for i = 1:np
		p = o[i]
		ϕ = sum([sigmoid_fast(-10*sdf(p, A_f[j], b_f[j], o_f[j], δ)) for j in 1:np+1])
		l += overlap * 1e-2 * max(ϕ - 1, 0)^2 / np
		for j = 1:nh
			for α ∈ [0.75, 0.5, 0.25]
				p = o[i] - α * A[i][j,:] .* b[i][j] / norm(A[i][j,:])^2
				l += expansion * -10 * min(0, p[2]) / (np * nh * length(α))
				ϕ = sum([sigmoid_fast(-10*sdf(p, A_f[j], b_f[j], o_f[j], δ)) for j in 1:np+1])
				l += overlap * 1e-2 * max(ϕ - 2, 0)^2 / (np * nh * length(α))
			end
		end
	end

	# spread
	for i = 1:np
		for j in setdiff(1:np, [i])
			l += expansion * 100 * 0.5 * (max(0, 0.5 - norm(o[i] - o[j])))^2 / (np^2)
		end
		l += expansion * -10 * min(0, o[i][2]) / np
	end
	return l
end
