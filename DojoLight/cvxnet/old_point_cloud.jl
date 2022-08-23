###############################################################################
# openGL point cloud
################################################################################
function point_cloud(vis::GLVisualizer.Visualizer, mechanism::Mechanism1170, z,
        p1, p2, resolution, eyeposition, lookat, up)
    set_mechanism!(vis, mechanism, z)
    point_cloud(vis, p1, p2, resolution, eyeposition, lookat, up)
end

function point_cloud(vis::GLVisualizer.Visualizer,
        p1, p2, resolution, eyeposition, lookat, up)

    GLVisualizer.set_camera!(vis;
        eyeposition=eyeposition,
        lookat=lookat,
        up=up)

    n1 = length(p1)
    n2 = length(p2)

    depth = zeros(Float32, resolution...)
    world_coordinates = zeros(3, n1 * n2)

    depth_buffer!(depth, glvis)
    depthpixel_to_world!(world_coordinates, depth, p1, p2, vis)

    return world_coordinates
end

function point_cloud_loss(vis::GLVisualizer.Visualizer, p1::Vector{Int}, p2::Vector{Int},
		resolution, eyeposition, lookat, up, WC, E, θ, polytope_dimensions)
	ne = length(E)
	np = length(polytope_dimensions)
	n1 = length(p1)
	n2 = length(p2)
	l = 0.0

	build_2d_convex_bundle!(vis, :root, θ, polytope_dimensions)
	WC_learned = [point_cloud(vis, p1, p2, resolution, e, lookat, up) for e in E]
	for i = 1:ne
		for j = 1:n1*n2
			l += 0.5 * norm(WC[i][2:3,j] - WC_learned[i][2:3,j])^2
		end
	end
	return l
end


################################################################################
# julia point cloud
################################################################################
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

# function julia_intersection(e::Vector, v::Vector, δ, A::Matrix{T}, b::Vector, o::Vector) where T
#     n = length(b)
#     eoff = e - o
#
# 	α = zeros(T, n) # ray length
# 	d = zeros(T, n) # distance to ray
#     for i = 1:n
# 		# @show i
#         denum = (A[i,:]' * v)
#         α[i] = (b[i] - A[i,:]' * eoff) / (denum + sign(denum)*1e-6)
#         x = eoff + α[i] .* v
#         d[i] = maximum(A * x .- b)
# 		d[i] = max(1e-3, d[i])
# 		# @show d[i]
#     end
#
# 	# αsoft = softmin(max.(0,α), δ)
# 	# dsoft = softmin(d, δ)
# 	# @show α
# 	# @show d
# 	# @show softweights(-α - exp(δ) * d, δ)
# 	# @show -α - exp(δ) * d
# 	w = softweights(-max.(0, α) - max(100, exp(δ)) * d, δ-4)
# 	αsoft = sum(w .* α)
# 	dsoft = sum(w .* d)
#     return αsoft, dsoft
# end
#
# function julia_intersection(e::Vector, v::Vector, δ, A::Vector{<:Matrix{T}},
#         b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}}) where T
#
#     np = length(b)
# 	α = zeros(T,np)
#     d = zeros(T,np)
#     for i = 1:np
#         n = length(b[i])
#         α[i], d[i] = julia_intersection(e, v, δ, A[i], b[i], o[i])
#     end
# 	w = softweights(-α - max(100, exp(δ)) * d, δ)
#     αsoft = sum(w .* α)
#     return αsoft
# end

function julia_point_cloud(e::Vector, β, δ, A::Vector{Matrix{T}}, b::Vector{<:Vector}, o::Vector{<:Vector}) where T
    nβ = length(β)
    P = zeros(T, 2, nβ)
    julia_point_cloud!(P, e, β, δ, A, b, o)
    return P
end

function julia_point_cloud!(P::Matrix, e::Vector, β, δ, A::Vector{<:Matrix}, b::Vector{<:Vector}, o::Vector{<:Vector})
	nβ = length(β)
    for i = 1:nβ
        v = [cos(β[i]), sin(β[i])]
        α = julia_intersection(e, v, δ, A, b, o)
        P[:,i] = e + α * v
    end
    return nothing
end

function julia_loss(P::Vector{<:Matrix}, e::Vector{<:Vector}, β::Vector, δ, θ::Vector{T},
        polytope_dimensions::Vector{Int}) where T

    ne = length(e)
    l = 0.0
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    for i = 1:ne
        Pθ = julia_point_cloud(e[i], β[i], δ, A, b, o)
        l += 0.5 * norm(P[i] - Pθ)^2 / size(Pθ, 2)
    end
    return l
end

function add_floor(θ, polytope_dimensions)
    A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    Af = [0 1.0]
    bf = [0.0]
    of = [0, 0.0]
    return pack_halfspaces([A..., Af], [b..., bf], [o..., of])
end

function add_floor(A, b, o)
    Af = [0.0 1.0]
    bf = [0.0]
    of = [0.0, 0.0]
    return [A..., Af], [b..., bf], [o..., of]
end


################################################################################
# visualize convex_bundle
################################################################################
function build_2d_convex_bundle!(vis::Visualizer, θ, polytope_dimensions;
        name::Symbol=:convex_bundle, color=RGBA(1,1,0,0.4))

    np = length(polytope_dimensions)
    A, b, o = unpack_halfspaces(θ, polytope_dimensions)

    for i = 1:np
		setobject!(
			vis[name][Symbol(i)][:center],
			HyperSphere(MeshCat.Point(0,o[i]...), 0.025),
			MeshPhongMaterial(color=RGBA(color.r, color.g, color.b, 1.0)))
        build_2d_polytope!(vis[name], A[i], b[i] + A[i]*o[i], name=Symbol(i), color=color)
    end
    return nothing
end

################################################################################
# visualize point cloud
################################################################################
function build_point_cloud!(vis::Visualizer, num_points::Vector{Int};
        color=RGBA(0.1,0.1,0.1,1), name::Symbol=:point_cloud)

    for (i,n) in enumerate(num_points)
        build_point_cloud!(vis[name], n, name=Symbol(i), color=color)
    end
    return nothing
end

function build_point_cloud!(vis::Visualizer, num_points::Int;
        name::Symbol=:point_cloud, fov_length=0.2,
        color=RGBA(0.1,0.1,0.1,1))

    for i = 1:num_points
        setobject!(
            vis[name][:eyeposition],
            HyperSphere(MeshCat.Point(0,0,0.0), 0.05),
            MeshPhongMaterial(color=color))
		setobject!(
			vis[name][:points][Symbol(i)],
			HyperSphere(MeshCat.Point(0,0,0.0), 0.025),
			MeshPhongMaterial(color=color))

        if i == 1 || i == num_points
            fov_name = Symbol(:fov_, i)
            build_segment!(vis[name]; color=color, name=fov_name)
        end
    end
    return nothing
end


function set_2d_point_cloud!(vis::Visualizer, e::Vector{<:Vector}, P::Vector{<:Matrix};
        name::Symbol=:point_cloud)

    ee = [[0; ei] for ei  in e]
    Pe = [[zeros(size(Pi,2))'; Pi] for Pi in P]
    set_point_cloud!(vis, ee, Pe, name=name)
    return nothing
end

function set_point_cloud!(vis::Visualizer, e::Vector{<:Vector}, P::Vector{<:Matrix};
        name::Symbol=:point_cloud)

    ne = length(e)
    for i = 1:ne
        set_point_cloud!(vis[name], e[i], P[i], name=Symbol(i))
    end
    return nothing
end

function set_point_cloud!(vis::Visualizer, e::Vector, P::Matrix;
        name::Symbol=:point_cloud, fov_length=0.2)

	num_points = size(P, 2)
    for i = 1:num_points
        settransform!(vis[name][:eyeposition], MeshCat.Translation(e...))
		settransform!(vis[name][:points][Symbol(i)], MeshCat.Translation(P[:,i]...))

        if i == 1 || i == num_points
            v = (P[:,i] - e)
            v = v ./ norm(v) * fov_length
            fov_name = Symbol(:fov_, i)
            set_segment!(vis[name], e, e + v; name=fov_name)
        end
    end
    return nothing
end
