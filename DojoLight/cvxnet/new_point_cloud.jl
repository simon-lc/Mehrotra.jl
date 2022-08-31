function pack_lp_parameters(e, v, ρ, A, b, o)
	return [e; v; ρ; vec(A); b; o]
end

function unpack_lp_parameters(parameters, num_cone::Int)
	off = 0
	e = parameters[off .+ (1:2)]; off += 2
	v = parameters[off .+ (1:2)]; off += 2
	ρ = parameters[off .+ (1:1)]; off += 1
	A = parameters[off .+ (1:2*num_cone)]; off += 2*num_cone
	A = reshape(A, (num_cone, 2))
	b = parameters[off .+ (1:num_cone)]; off += num_cone
	o = parameters[off .+ (1:2)]; off += 2
	return e, v, ρ, A, b, o
end

function lp_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_cone = length(duals)

	e, v, ρ, A, b, o = unpack_lp_parameters(parameters, num_cone)
	α = y[1]
	x = y[2:3]

    res = [
		1 + 1/ρ[1] * v' * (e + α * v - x);
		1/ρ[1] * (x - (e + α*v)) + A'*z;
		s - (b - A * (x - o))
        # z .* s .- κ[1];
        ]
    return res
end

function lp_solver(num_cone::Int; ρ=1e-2)
	e = zeros(2)
	v = zeros(2)
	A = zeros(num_cone, 2)
	b = zeros(num_cone)
	o = zeros(2)

	parameters = pack_lp_parameters(e, v, ρ, A, b, o)
	num_primals = 3
	solver = Mehrotra.Solver(lp_residual, num_primals, num_cone,
		parameters=parameters,
		options=Options(
			residual_tolerance=ρ/10,
			complementarity_tolerance=ρ,
			verbose=false,
			),
		)
	return solver
end

function soft_intersection(e::Vector, v::Vector, ρ, A::Matrix, b::Vector, o::Vector;
		solver::Solver=lp_solver(length(b), ρ=ρ))

	parameters = pack_lp_parameters(e, v, ρ, A, b, o)
	set_parameters!(solver, parameters)
	solve!(solver)
	sol = deepcopy(solver.solution.primals)
	∂sol = deepcopy(solver.data.solution_sensitivity[1:3,6:end])
    return sol, ∂sol
end

function soft_intersection(e::Vector, v::Vector, ρ, A::Vector{<:Matrix{T}},
        b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in length.(b)]) where T

    np = length(b)
	nθ = 3length.(b) .+ 2

	sols = zeros(T,3np)
	∂sols = zeros(T,3,sum(nθ))
	off = 0
    for i = 1:np
		sols[3*(i-1) .+ (1:3)], ∂sols[:,off .+ (1:nθ[i])] = soft_intersection(e, v, ρ, A[i], b[i], o[i], solver=solvers[i])
		off += nθ[i]
    end
	return sols, ∂sols
end

function new_point_cloud(e::Vector, β, ρ, A::Vector{<:Matrix{T}},
        b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in length.(b)]) where T

	nβ = length(β)
	d = zeros(2,nβ)

	off = 0
	for i = 1:nβ
		v = [cos(β[i]), sin(β[i])]
		sols = soft_intersection(e, v, ρ, A, b, o, solvers=solvers)[1]
		d[:,i] = new_point_cloud(e, v, ρ, sols)
	end
	return d
end

function new_point_cloud(e::Vector, v::Vector, ρ, sols::Vector{T}) where T
	np = Int(length(sols) / 3)
	α = zeros(T,np)
	δ = zeros(T,np)

	off = 0
	for i = 1:np
		α[i] = sols[off + 1]; off += 1
		x = sols[off .+ (1:2)]; off += 2
		δ[i] = 0.5 * (e + α[i] * v - x)' * (e + α[i] * v - x)
    end
	values = - (α + δ ./ ρ) # optimization objective function
	αsoft = sum(α .* softweights(values, -4 - log(ρ) - log(ρ/1e-2)))
	d = e + αsoft * v
	return d
end

function point_loss(e::Vector, v::Vector, ρ, sols::Vector{T}, d̂::Vector) where T
	np = Int(length(sols) / 3)
	α = zeros(T,np)
	δ = zeros(T,np)

	off = 0
	for i = 1:np
		α[i] = sols[off + 1]; off += 1
		x = sols[off .+ (1:2)]; off += 2
		δ[i] = 0.5 * (e + α[i] * v - x)' * (e + α[i] * v - x)
    end
	values = - (α + δ ./ ρ) # optimization objective function
	αsoft = sum(α .* softweights(values, -4 - log(ρ) - log(ρ/1e-2)))
	d = e + αsoft * v
	return 0.5 * (d - d̂)' * (d - d̂)
end

function point_loss(e::Vector, v::Vector, ρ, θ::Vector{T}, polytope_dimensions::Vector{Int}, d̂::Vector;
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in polytope_dimensions]) where T

	np = length(polytope_dimensions)
	nθ = 3*polytope_dimensions .+ 2
	g = zeros(sum(nθ))
	H = zeros(sum(nθ), sum(nθ))
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)

	sols, ∂sols = soft_intersection(e, v, ρ, A, b, o, solvers=solvers)
	l = point_loss(e, v, ρ, sols, d̂)
	∂l∂sols = ForwardDiff.gradient(sols -> point_loss(e, v, ρ, sols, d̂), sols)
	∂l∂2sols = ForwardDiff.hessian(sols -> point_loss(e, v, ρ, sols, d̂), sols)

	off = 0
	for i = 1:np
		g[off .+ (1:nθ[i])] .= ∂sols[:,off .+ (1:nθ[i])]' * ∂l∂sols[3*(i-1) .+ (1:3)]; off += nθ[i]
	end

	offi = 0
	for i = 1:np
		Di = ∂sols[:,offi .+ (1:nθ[i])]
		offj = 0
		for j = 1:np
			Dj = ∂sols[:,offj .+ (1:nθ[j])]
			Sij = ∂l∂2sols[3*(i-1) .+ (1:3), 3*(j-1) .+ (1:3)]
			H[offi .+ (1:nθ[i]), offj .+ (1:nθ[j])] .= Di' * Sij * Dj
			offj += nθ[j]
		end
		offi += nθ[i]
	end
	return l, g, H
end

function soft_loss(d::Vector{<:Matrix}, e::Vector{<:Vector}, β::Vector, ρ, θ::Vector,
		polytope_dimensions::Vector{Int};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in polytope_dimensions])

    ne = length(e)
    l = 0.0
	g = zeros(length(θ))
	H = zeros(length(θ), length(θ))
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    for i = 1:ne
		for j = 1:length(β[i])
			v = [cos(β[i][j]), sin(β[i][j])]
			lj, gj, Hj = point_loss(e[i], v, ρ, θ, polytope_dimensions, d[i][:,j], solvers=solvers)
			l += lj
			g += gj
			H += Hj
		end
    end
    return l, g, H
end




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
op1 = [0.2, 0.2]

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
op2 = [-0.5, 0.2]

Af = [0.0  1.0]
bf = [0.0]
of = [0.0, 0.0]

AA = [Ap1, Ap2, Ap, Af]
bb = [bp1, bp2, bp, bf]
oo = [op1, op2, op, of]

op = [0.0, +0.5]
ep = [0.0, +3.0]
vp = [0.0, -1.0]
cp = 1e-3
d̂ = [2,1.0]

parameters = pack_lp_parameters(ep, vp, cp, Ap, bp, op)
solver = lp_solver(num_cone, ρ=cp)
set_parameters!(solver, parameters)
solve!(solver)


setobject!(vis[:camera], HyperSphere(MeshCat.Point(0, ep...), 0.05))
build_2d_polytope!(vis[:polytope], Ap, bp + Ap * op, name=:poly, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope1], Ap1, bp1 + Ap1 * op1, name=:poly1, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope1], Ap2, bp2 + Ap2 * op2, name=:poly2, color=RGBA(0,0,0,0.3))
setobject!(vis[:intersection],
	HyperSphere(MeshCat.Point(0, (ep + αr * vp)...), 0.01))


β = Vector(range(-0.1π, -0.9π, length=100))
d = new_point_cloud(ep, β, cp, AA, bb, oo)
num_points = size(d, 2)
build_point_cloud!(vis[:point_cloud], num_points; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
set_2d_point_cloud!(vis, [ep], [d]; name=:point_cloud)




sols, ∂sols = soft_intersection(ep, vp, cp, AA, bb, oo)
point_loss(ep, vp, cp, sols, d̂)
AA = [Ap1, Ap2, Ap, Af]
bb = [bp1, bp2, bp, bf]
oo = [op1, op2, op, of]
θ, polytope_dimensions = pack_halfspaces(AA, bb, oo)
θ += 0.1rand(length(θ))
solvers = [lp_solver(l, ρ=cp) for l in polytope_dimensions]
l, g, H = point_loss(ep, vp, cp, θ, polytope_dimensions, d̂, solvers=solvers)
g0 = FiniteDiff.finite_difference_gradient(θ -> point_loss(ep, vp, cp, θ, polytope_dimensions, d̂, solvers=solvers)[1], θ)
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
    if di[2] > 2e-1
        push!(d_object, di)
    end
end
d_object
d_object = hcat(d_object...)

# convex bundle parameterization
nh = 20
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
# solvers = [lp_solver(l, ρ=cp) for l in polytope_dimensions]
# soft_loss([d], [ep], [β], cp, θ, polytope_dimensions, solvers=solvers)

θ_f, polytope_dimensions_f = add_floor(θinit, polytope_dimensions)
solvers_f = [lp_solver(l, ρ=cp) for l in polytope_dimensions_f]

function local_loss(θ)
	θ_f, polytope_dimensions_f = add_floor(θ, polytope_dimensions)
	l = soft_loss([d], [ep], [β], cp, θ_f, polytope_dimensions_f, solvers=solvers_f)[1]
end
function local_grad(θ)
	nθ = length(θ)
	θ_f, polytope_dimensions_f = add_floor(θ, polytope_dimensions)
	soft_loss([d], [ep], [β], cp, θ_f, polytope_dimensions_f, solvers=solvers_f)[2][1:nθ]
end
function local_hess(θ)
	nθ = length(θ)
	θ_f, polytope_dimensions_f = add_floor(θ, polytope_dimensions)
	soft_loss([d], [ep], [β], cp, θ_f, polytope_dimensions_f, solvers=solvers_f)[3][1:nθ,1:nθ]
end

################################################################################
# projection
################################################################################
# θmin = vcat([[-Inf * ones(nh); +0.05 * ones(nh); -Inf * ones(2)] for i=1:np]...)
θmin = vcat([[-Inf * ones(2nh); +0.05 * ones(nh); -Inf * ones(2)] for i=1:np]...)
# θmax = vcat([[+Inf * ones(nh); +Inf  * ones(nh); +Inf * ones(2)] for i=1:np]...)
θmax = vcat([[+Inf * ones(2nh); +Inf  * ones(nh); +Inf * ones(2)] for i=1:np]...)
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
        max_iterations=10,
        reg_min=1e-4,
        reg_max=1e+1,
        reg_step=2.0,
        line_search_iterations=5,
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
