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


local_loss(θ) = shape_loss(θ, ep, β, cp, d0,
	δ=100.0,
	altitude_threshold=0.1,
	rendering=10.0,
	sdf_matching=1.0,
	overlap=0.1,
	individual=1.0,
	expansion=0.5,
	side_regularization=1.0,
	)
local_grad(θ) = ForwardDiff.gradient(θ -> local_loss(θ), θ)
local_hess(θ) = Diagonal(1e-3*ones(length(θ)))

local_loss(θinit)
local_loss(θiter[end])
local_grad(θinit)
local_hess(θinit)

# projection
local_projection(θ) = projection(θ, polytope_dimensions)
# clamping
local_clamping(Δθ) = clamping(Δθ, polytope_dimensions)

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

visualize_iterates!(vis, θiter, polytope_dimensions, ep, β, cp)

# AAt, bbt, oot = unpack_halfspaces(θiter[end], polytope_dimensions)
# # inside sampling, overlap penalty
# for i = 1
# 	p = oot[i]
# 	for j = 1:length(bbt[i])
# 		p = oot[i] + 1.0 * AAt[i][j,:] .* bbt[i][j] / norm(AAt[i][j,:])^2
# 		setobject!(vis[:sampling][Symbol(i)][Symbol(j)], HyperSphere(MeshCat.Point(0, p...), 0.05),
# 			MeshPhongMaterial(color=RGBA(1,0,0,1)))
# 		sleep(0.1)
# 	end
# end
