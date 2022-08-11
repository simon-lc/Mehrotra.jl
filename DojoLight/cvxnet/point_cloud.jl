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
		resolution, eyeposition, lookat, up, WC, E, θ, bundle_dimensions)
	ne = length(E)
	np = length(bundle_dimensions)
	n1 = length(p1)
	n2 = length(p2)
	l = 0.0

	build_2d_convex_bundle!(vis, :root, θ, bundle_dimensions)
	WC_learned = [point_cloud(vis, p1, p2, resolution, e, lookat, up) for e in E]
	for i = 1:ne
		for j = 1:n1*n2
			l += 0.5 * norm(WC[i][2:3,j] - WC_learned[i][2:3,j])^2
		end
	end
	# for i = 1:np
	# 	delete!(glvis, Symbol(:polytope_, i))
	# end
	return l
end

function sumeet_intersection(e::Vector, v::Vector, A::Matrix, b::Vector, o::Vector, δ)
    n = length(b)
    eoff = e - o
    αmin = +Inf

    for i = 1:n
        denum = (A[i,:]' * v)
        (abs(denum) < 1e-3) && continue
        α = (b[i] - A[i,:]' * eoff) / denum
        x = eoff + α * v
        s = maximum(A * x - b)
        (s <= 1e-10) && (αmin = min(αmin, α))
    end

	# add smoothness correction
	if αmin < +Inf && length(b) > 1
		x = eoff + αmin * v
		c = A * x - b
		cmax, imax = findmax(c)
		c = c[setdiff(1:length(c), imax)]
		cmax2, imax2 = findmax(c)
		αmin += 1/δ^2 * exp(sqrt(δ) * (cmax2 - cmax))
	end
    return αmin
end

function sumeet_intersection(e::Vector, v::Vector, A::Vector{<:Matrix{T}},
        b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}}, δ) where T

    np = length(b)
    α = zeros(T,np)
    for i = 1:np
        n = length(b[i])
        α[i] = sumeet_intersection(e, v, A[i], b[i], o[i], δ)
    end
    αsoft = -softmax(-α, δ)
    return αsoft
end

function sumeet_point_cloud(e::Vector, β, θ::Vector{T},
        bundle_dimensions::Vector{Int}, δ) where T

    nβ = length(β)
    P = zeros(T, 2, nβ)
    sumeet_point_cloud!(P, e, β, θ, bundle_dimensions, δ)
    return P
end

function sumeet_point_cloud!(P::Matrix, e::Vector, β, θ::Vector{T},
        bundle_dimensions::Vector{Int}, δ) where T

    nβ = length(β)
    A, B, o = unpack_halfspaces(θ, bundle_dimensions)
    for i = 1:nβ
        v = [cos(β[i]), sin(β[i])]
        α = sumeet_intersection(e, v, A, B, o, δ)
        P[:,i] = e + α * v
    end
    return nothing
end

function sumeet_loss(P::Vector{<:Matrix}, e::Vector{<:Vector}, β::Vector, θ::Vector{T},
        bundle_dimensions::Vector{Int}, δ) where T

    ne = length(e)
    l = 0.0
    for i = 1:ne
        Pθ = sumeet_point_cloud(e[i], β[i], θ, bundle_dimensions, δ)
        l += 0.5 * norm(P[i] - Pθ)^2
    end
    return l
end

################################################################################
# visualize point cloud
################################################################################
function build_2d_point_cloud!(vis::Visualizer, P::Vector{<:Matrix}, e::Vector{<:Vector};
        color=RGBA(0.1,0.1,0.1,1), name::Symbol=:point_cloud)

    ee = [[0; ei] for ei  in e]
    Pe = [[zeros(size(Pi,2))'; Pi] for Pi in P]
    build_point_cloud!(vis, Pe, ee, color=color, name=name)
    return nothing
end

function build_point_cloud!(vis::Visualizer, P::Vector{<:Matrix}, e::Vector{<:Vector};
        color=RGBA(0.1,0.1,0.1,1), name::Symbol=:point_cloud)

    ne = length(e)
    for i = 1:ne
        build_point_cloud!(vis[name], P[i], e[i], name=Symbol(i), color=color)
    end
    return nothing
end

function build_point_cloud!(vis::Visualizer, P::Matrix, e::Vector;
        name::Symbol=:point_cloud, fov_length=0.2,
        color=RGBA(0.1,0.1,0.1,1))

    for i = 1:nβ
        setobject!(
            vis[name][:eyeposition],
            HyperSphere(MeshCat.Point(0,0,0.0), 0.05),
            MeshPhongMaterial(color=color))
        settransform!(vis[name][:eyeposition], MeshCat.Translation(e...))

        if i == 1 || i == nβ
            v = (P[:,i] - e)
            v = v ./ norm(v) * fov_length
            fov_name = Symbol(:fov_, i)
            build_segment!(vis[name]; color=color, name=fov_name)
            set_segment!(vis[name], e, e + v; name=fov_name)
        end

        setobject!(
            vis[name][Symbol(i)],
            HyperSphere(MeshCat.Point(0,0,0.0), 0.025),
            MeshPhongMaterial(color=color))
        settransform!(vis[name][Symbol(i)], MeshCat.Translation(P[:,i]...))
    end
    return nothing
end


function build_2d_convex_bundle!(vis::Visualizer, θ, bundle_dimensions;
        name::Symbol=:convex_bundle, color=RGBA(1,1,0,0.4))

    np = length(bundle_dimensions)
    A, b, o = unpack_halfspaces(θ, bundle_dimensions)

    for i = 1:np
		setobject!(
			vis[name][Symbol(i)][:center],
			HyperSphere(MeshCat.Point(0,o[i]...), 0.025),
			MeshPhongMaterial(color=RGBA(color.r, color.g, color.b, 1.0)))
        build_2d_polytope!(vis[name], A[i], b[i] + A[i]*o[i], name=Symbol(i), color=color)
    end
    return nothing
end
