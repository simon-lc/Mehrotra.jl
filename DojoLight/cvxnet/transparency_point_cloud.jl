function trans_intersection(e::Vector, v::Vector, ρ, A::Matrix, b::Vector, o::Vector)
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
	return 0.5 * (d - d̂)' * (d - d̂)
end

function trans_point_loss(e::Vector, v::Vector, ρ, θ::Vector{T},
		polytope_dimensions::Vector{Int}, d̂::Vector) where T

	np = length(polytope_dimensions)
	nθ = 3*polytope_dimensions .+ 2
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
		for j = 1:length(β[i])
			v = [cos(β[i][j]), sin(β[i][j])]
			lj = trans_point_loss(e[i], v, ρ, θ, polytope_dimensions, d[i][:,j])
			l += lj
		end
    end
    return l
end

Main.@profiler [local_loss(θinit) for i=1:1000]


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

include("../src/DojoLight.jl")
colors = [
    RGBA(1,0,0,1),
    RGBA(0,1,0,1),
    RGBA(0,0,1,1),
    RGBA(0,1,1,1),
    RGBA(1,1,0,1),
    RGBA(1,0,1,1),
    RGBA(0.5,0.5,0.5,1),
];

vis = Visualizer()
render(vis)




# parameters
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2)
bp = 0.5*[
    +1,
    +1,
    +1,
     1,
    ]
op = [0.0, +0.5]

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

AA = [Ap1, Ap2, Ap, Af]
bb = [bp1, bp2, bp, bf]
oo = [op1, op2, op, of]


AA = [Ap1, Ap2, Ap, Af]
bb = [bp1, bp2, bp, bf]
oo = [op1, op2, op, of]

d = trans_point_cloud(ep, [-π/2], cp, AA, bb, oo)
d = trans_point_cloud(ep, β, cp, AA, bb, oo)

op = [0.0, +0.5]
ep = [0.0, +2.0]
vp = [0.0, -1.0]
cp = 1/0.05
d̂ = [2,1.0]

parameters = pack_lp_parameters(ep, vp, cp, Ap, bp, op)
solver = lp_solver(num_cone, ρ=cp)
set_parameters!(solver, parameters)
solve!(solver)


build_2d_polytope!(vis[:polytope], Ap, bp + Ap * op, name=:poly, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope], Ap1, bp1 + Ap1 * op1, name=:poly1, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope], Ap2, bp2 + Ap2 * op2, name=:poly2, color=RGBA(0,0,0,0.3))


β = Vector(range(-0.2π, -0.8π, length=100))
d = trans_point_cloud(ep, β, cp, AA, bb, oo)
num_points = size(d, 2)
build_point_cloud!(vis[:point_cloud], num_points; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
set_2d_point_cloud!(vis, [ep], [d]; name=:point_cloud)


build_point_cloud!(vis[:point_cloud], num_points; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
anim = MeshCat.Animation(20)
for (i, cpi) in enumerate(range(0, 3.0, length=100))
	atframe(anim, i) do
		d = trans_point_cloud(ep, β, exp(log(10)*cpi), AA, bb, oo)
		num_points = size(d, 2)
		set_2d_point_cloud!(vis, [ep], [d]; name=:point_cloud)
	end
end
setanimation!(vis, anim)
open(vis)
set_floor!(vis)
set_background!(vis)
set_light!(vis)



sols, ∂sols = soft_intersection(ep, vp, cp, AA, bb, oo)
point_loss(ep, vp, cp, sols, d̂)
AA = [Ap1, Ap2, Ap, Af]
bb = [bp1, bp2, bp, bf]
oo = [op1, op2, op, of]
θ, polytope_dimensions = pack_halfspaces(AA, bb, oo)
θ += 0.1rand(length(θ))
solvers = [lp_solver(l, ρ=cp) for l in polytope_dimensions]
l = trans_point_loss(ep, vp, cp, θ, polytope_dimensions, d̂)
g0 = FiniteDiff.finite_difference_gradient(θ -> trans_point_loss(ep, vp, cp, θ, polytope_dimensions, d̂), θ)
norm(g - g0)



################################################################################
# initialization
################################################################################
# point cloud reaching the object
d_object = []
ne = 1
d
nβ = length(β)
for j = 1:nβ
    di = d[:,j]
    if di[2] > 3e-1
        push!(d_object, di)
    end
end
d_object
d_object = hcat(d_object...)

# convex bundle parameterization
nh = 10
polytope_dimensions = [nh, nh, nh, nh, nh]
np = length(polytope_dimensions)

# k-mean clustering
kmres = kmeans(d_object, np)
# display k-mean result
for i = 1:size(d_object, 2)
    ik = kmres.assignments[i]
    setobject!(
        vis[:cluster][Symbol(ik)][Symbol(i)],
        HyperSphere(MeshCat.Point(0,0,0.0), 0.035),
        MeshPhongMaterial(color=colors[ik]))
    settransform!(vis[:cluster][Symbol(ik)][Symbol(i)], MeshCat.Translation(0.2, d_object[:,i]...))
end
for i = 1:np
    setobject!(
        vis[:cluster][Symbol(i)][:center],
        HyperRectangle(MeshCat.Vec(-0.05,-0.05,-0.05), MeshCat.Vec(0.1,0.1,0.1)),
        MeshPhongMaterial(color=colors[i]))
    settransform!(vis[:cluster][Symbol(i)][:center], MeshCat.Translation(0.2, kmres.centers[:,i]...))
end
# initialization
b_char = 2 * mean(sqrt.(kmres.costs))
θinit = zeros(0)
for i = 1:np
    angles = range(-π, π, length=nh+1)[1:end-1] + 0.15*rand(nh)
    # θi = [range(-π, π, length=nh+1)[1:end-1] + 0.15*rand(nh); b_char*ones(nh); kmres.centers[:, i]]
    θi = [vcat([[cos(a), sin(a)] for a in angles]...); b_char*ones(nh); kmres.centers[:, i]]
    A, b, o = unpack_halfspaces(θi)
    push!(θinit, pack_halfspaces(A, b, o)...)
end
θinit
build_2d_convex_bundle!(vis, θinit, polytope_dimensions, name=:initial, color=RGBA(1,1,0,0.4))


θdiag = zeros(0)
for i = 1:np
    # θi = [1e-2 * ones(2nh); 1e-0 * ones(nh); 1e-0 * ones(2)]
    θi = [1e-0 * ones(2nh); 1e+1 * ones(nh); 1e+1 * ones(2)]
    A, b, o = unpack_halfspaces(θi)
    push!(θdiag, pack_halfspaces(A, b, o)...)
end
θdiag


# θ, polytope_dimensions = pack_halfspaces(AA, bb, oo)

Ainit, binit, oinit = unpack_halfspaces(deepcopy(θinit), polytope_dimensions)
function local_loss(θ)
	θ_f, polytope_dimensions_f = add_floor(θ, polytope_dimensions)
	l = trans_point_loss([ep], [β], cp, θ, polytope_dimensions, [d])
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
	l += 1.0 * sum([0.5*norm(bi .- 1.0)^2 for bi in b])
	l += 0.1 * sum([0.5*norm(A[i] - Ainit[i])^2 for i=1:np])
	l += 0.1 * sum([0.5*norm(o[i] - oinit[i])^2 for i=1:np])
end
local_grad(θ) = ForwardDiff.gradient(θ -> local_loss(θ), θ)
# local_hess(θ) = ForwardDiff.hessian(θ -> local_loss(θ), θ)
local_hess(θ) = I(length(θ))

local_loss(θinit)
local_grad(θinit)
local_hess(θinit)

################################################################################
# projection
################################################################################
θmin = vcat([[-1.0 * ones(2nh); +0.05 * ones(nh); -3.0 * ones(2)] for i=1:np]...)
θmax = vcat([[+1.0 * ones(2nh); +0.40 * ones(nh); +3.0 * ones(2)] for i=1:np]...)
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
Δθmin = vcat([[-0.30 * ones(2nh); -0.025 * ones(nh); -0.025 * ones(2)] for i=1:np]...)
Δθmax = vcat([[+0.30 * ones(2nh); +0.025 * ones(nh); +0.025 * ones(2)] for i=1:np]...)
function clamping(Δθ, Δθmin, Δθmax, polytope_dimensions)
    return clamp.(Δθ, Δθmin, Δθmax)
end
local_clamping(Δθ) = clamping(Δθ, Δθmin, Δθmax, polytope_dimensions)

################################################################################
# solve
################################################################################
θsol, θiter = newton_solver!(θinit, local_loss, local_grad, local_hess, local_projection, local_clamping;
        max_iterations=40,
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

anim = MeshCat.Animation(10)
for i = 1:T
    atframe(anim, i) do
        for ii = 1:T
            setvisible!(vis[:iterates][Symbol(ii)], ii == i)
        end
    end
end

setanimation!(vis, anim)
