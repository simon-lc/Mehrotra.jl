
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
