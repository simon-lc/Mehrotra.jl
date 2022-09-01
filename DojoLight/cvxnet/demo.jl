using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays
using ForwardDiff
using Clustering
using LinearAlgebra

include("../src/DojoLight.jl")
include("../system_identification/newton_solver.jl")
include("halfspace.jl")
include("transparency_point_cloud.jl")
include("visuals.jl")
include("softmax.jl")
include("utils.jl")

vis = Visualizer()
open(vis)


# parameters
Ap0 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2)
bp0 = 0.5*[
    +1,
    +1,
    +1,
     1,
    ]
op0 = [0.0, +0.5]

Ap1 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.30ones(4,2)
bp1 = 0.25*[
    +1,
    +1,
    +1,
     1,
    ]
op1 = [0.5, 0.2]

Ap2 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bp2 = 0.25*[
    +1,
    +1,
    +1,
     1,
    ]
op2 = [-0.7, 0.2]

Af = [0.0  1.0]
bf = [0.0]
of = [0.0, 0.0]

AA = [Ap0, Ap1, Ap2, Af]
bb = [bp0, bp1, bp2, bf]
oo = [op0, op1, op2, of]

op = [0.0, +0.5]
ep = [0.0, +2.0]
vp = [0.0, -1.0]
cp = 1/0.01
β = Vector(range(-0.2π, -0.8π, length=100))
nβ = length(β)

build_2d_polytope!(vis[:polytope], Ap0, bp0 + Ap0 * op0, name=:poly0, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope], Ap1, bp1 + Ap1 * op1, name=:poly1, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope], Ap2, bp2 + Ap2 * op2, name=:poly2, color=RGBA(0,0,0,0.3))

d0 = trans_point_cloud(ep, β, cp*100, AA, bb, oo)
num_points = size(d0, 2)
build_point_cloud!(vis[:point_cloud], num_points; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
set_2d_point_cloud!(vis, [ep], [d0]; name=:point_cloud)



nh = 8
polytope_dimensions = [nh,nh,nh,nh,nh,nh]
np = length(polytope_dimensions)
θinit, d_object, kmres = parameter_initialization(d0, polytope_dimensions; altitude_threshold=0.3)
Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
visualize_kmeans!(vis, θinit, polytope_dimensions, d_object, kmres)
polytope_dimensions

θdiag = zeros(0)
for i = 1:np
	θi = [1e-1 * ones(2nh); 1e+0 * ones(nh); 1e-0 * ones(2)]
    # θi = [1e-2 * ones(nh); 1e+1 * ones(nh); 1e+0 * ones(2)]
    A, b, o = unpack_halfspaces(θi)
    push!(θdiag, pack_halfspaces(A, b, o)...)
end
θdiag



function local_loss(θ)
	δ = 100.0
	altitude_threshold = 0.1

	rendering = 10.0
	sdf_matching = 1.0
	overlap = 0.1
	individual = 1.0
	expansion = 0.3
	# regularization = 0.001
	side_regularization = 2.0
	# norm_regularization = 0.01

	θ_f, polytope_dimensions_f = add_floor(θ, polytope_dimensions)
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
	A_f, b_f, o_f = unpack_halfspaces(θ_f, polytope_dimensions_f)

	l = rendering * trans_point_loss([ep], [β], cp, θ_f, polytope_dimensions_f, [d0])

	# regularization
	# l += regularization * 1.0 * sum([0.5*norm(A[i] - Ainit[i])^2 + 0.5 * norm(A[i] - Ainit[i]) for i=1:np]) / (np * nh)
	l += side_regularization * 10.0 * sum([0.5*norm(bi .- 0.50)^2 for bi in b]) / (np * nh)
	# l += regularization * 1.0 * sum([0.5*norm(o[i] - oinit[i])^2 for i=1:np]) / np

	# # normalization
	# for Ai in A
	# 	for j = 1:size(Ai,1)
	# 		l += norm_regularization * 10.0 * 0.5 * (norm(Ai[j,:]) .- 1)^2 / (np * nh)
	# 	end
	# end

	# sdf matching
	for i = 1:nβ
		p = d0[:,i]
		ϕ = sdf(p, A_f, b_f, o_f, δ)
		l += sdf_matching * 0.1 * (ϕ^2 + abs(ϕ)) / nβ
	end

	# individual
	for i = 1:nβ
		di = d0[:,i]
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






sigmoid_fast(-10.0)

local_grad(θ) = ForwardDiff.gradient(θ -> local_loss(θ), θ)
# local_hess(θ) = ForwardDiff.hessian(θ -> local_loss(θ), θ)
local_hess(θ) = Diagonal(1e-3*ones(length(θ)))

local_loss(θinit)
# @benchmark local_loss(θinit)
local_loss(θiter[end])
local_grad(θinit)
# @benchmark local_grad(θinit)
# Main.@profiler [local_loss(θinit) for i = 1:100]
# Main.@profiler local_grad(θinit)
local_hess(θinit)

################################################################################
# projection
################################################################################
θmin = vcat([[-1.0 * ones(2nh); +0.05 * ones(nh); -3.0 * ones(2)] for i=1:np]...)
# θmin = vcat([[-1000.0 * ones(nh); +0.05 * ones(nh); -3.0 * ones(2)] for i=1:np]...)
θmax = vcat([[+1.0 * ones(2nh); +0.40 * ones(nh); +3.0 * ones(2)] for i=1:np]...)
# θmax = vcat([[+1000.0 * ones(nh); +0.40 * ones(nh); +3.0 * ones(2)] for i=1:np]...)
function projection(θ, θmin, θmax, polytope_dimensions)
    A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    np = length(polytope_dimensions)
    for i = 1:np
        for j = 1:size(A[i],1)
            A[i][j,:] = A[i][j,:] / (1e-6 + norm(A[i][j,:]))
        end
    end
    return clamp.(θ, θmin, θmax)
end
local_projection(θ) = projection(θ, θmin, θmax, polytope_dimensions)

################################################################################
# clamping
################################################################################
Δθmin = vcat([[-0.60 * ones(2nh); -0.05 * ones(nh); -0.05 * ones(2)] for i=1:np]...)
# Δθmin = vcat([[-2.60 * ones(nh); -0.05 * ones(nh); -0.05 * ones(2)] for i=1:np]...)
Δθmax = vcat([[+0.60 * ones(2nh); +0.05 * ones(nh); +0.05 * ones(2)] for i=1:np]...)
# Δθmax = vcat([[+2.60 * ones(nh); +0.05 * ones(nh); +0.05 * ones(2)] for i=1:np]...)
function clamping(Δθ, Δθmin, Δθmax, polytope_dimensions)
    return clamp.(Δθ, Δθmin, Δθmax)
end
local_clamping(Δθ) = clamping(Δθ, Δθmin, Δθmax, polytope_dimensions)

################################################################################
# solve
################################################################################
θsol, θiter = newton_solver!(θinit, local_loss, local_grad, local_hess, local_projection, local_clamping;
        max_iterations=30,
        reg_min=1e-4,
        reg_max=1e+1,
        reg_step=2.0,
        line_search_iterations=10,
        residual_tolerance=1e-4,
        D=Diagonal(θdiag))


T = length(θiter)
build_2d_convex_bundle!(vis, θinit, polytope_dimensions, name=:initial)
for i = 1:T
	build_2d_convex_bundle!(vis[:iterates], θiter[i], polytope_dimensions, name=Symbol(i))
end
build_point_cloud!(vis[:iterates][:point_cloud], num_points;
	color=RGBA(0.8,0.1,0.1,1), name=Symbol(1))


anim = MeshCat.Animation(10)
for i = 1:T
    atframe(anim, i) do
		θ_f, polytope_dimensions_f = add_floor(θiter[i], polytope_dimensions)
		d = trans_point_cloud(ep, β, cp, unpack_halfspaces(θ_f, polytope_dimensions_f)...)
		set_2d_point_cloud!(vis[:iterates], [ep], [d]; name=:point_cloud)
        for ii = 1:T
            setvisible!(vis[:iterates][Symbol(ii)], ii == i)
        end
    end
end
setanimation!(vis, anim)


AAt, bbt, oot = unpack_halfspaces(θiter[end], polytope_dimensions)
# inside sampling, overlap penalty
for i = 1
	p = oot[i]
	for j = 1:length(bbt[i])
		p = oot[i] + 1.0 * AAt[i][j,:] .* bbt[i][j] / norm(AAt[i][j,:])^2
		setobject!(vis[:sampling][Symbol(i)][Symbol(j)], HyperSphere(MeshCat.Point(0, p...), 0.05),
			MeshPhongMaterial(color=RGBA(1,0,0,1)))
		sleep(0.1)
	end
end
# open(vis)
