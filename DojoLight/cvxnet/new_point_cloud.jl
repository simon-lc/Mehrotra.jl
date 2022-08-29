using GeometryBasics
using Plots
using RobotVisualizer
using MeshCat
using Polyhedra
using Quaternions
using Optim
using StaticArrays

include("../src/DojoLight.jl")

vis = Visualizer()
render(vis)

function pack_lp_parameters(e, v, A, b, o, ρ)
	return [e; v; vec(A); b; o; ρ]
end

function unpack_lp_parameters(parameters, num_cone::Int)
	off = 0
	e = parameters[off .+ (1:2)]; off += 2
	v = parameters[off .+ (1:2)]; off += 2
	A = parameters[off .+ (1:2*num_cone)]; off += 2*num_cone
	A = reshape(A, (num_cone, 2))
	b = parameters[off .+ (1:num_cone)]; off += num_cone
	o = parameters[off .+ (1:2)]; off += 2
	ρ = parameters[off .+ (1:1)]; off += 1
	return e, v, A, b, o, ρ
end

function lp_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_cone = length(duals)

	e, v, A, b, o, ρ = unpack_lp_parameters(parameters, num_cone)
	yr = y[1]
	yp = y[2]

    res = [
		yr + z'*A*v;
		1/ρ[1] * yp - z'*ones(num_cone);
		s - (yp * ones(num_cone) + b - A * (e + yr * v - o))
        # z .* s .- κ[1];
        ]
    return res
end

# function lp_residual(primals, duals, slacks, parameters)
#     y, z, s = primals, duals, slacks
#     num_cone = length(duals)
#
# 	e, v, A, b, o, ρ = unpack_lp_parameters(parameters, num_cone)
# 	yr = y[1]
# 	yp = y[2]
#
#     res = [
# 		yr + z'*A*v;
# 		1/ρ[1] * yp - z'*ones(num_cone);
# 		s - (yp * ones(num_cone) + b - A * (e + yr * v - o))
#         # z .* s .- κ[1];
#         ]
#     return res
# end

function lp_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_cone = length(duals)

	e, v, A, b, o, ρ = unpack_lp_parameters(parameters, num_cone)
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

	parameters = pack_lp_parameters(e, v, A, b, o, ρ)
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

	parameters = pack_lp_parameters(e, v, A, b, o, ρ)
	set_parameters!(solver, parameters)
	solve!(solver)
	αr = solver.solution.primals[1]
	x = solver.solution.primals[2:3]
	αp = norm(e + αr * v - x)^2 #+ 0.5norm(e + αr * v - x)
    return αr, αp
end

function soft_intersection(e::Vector, v::Vector, ρ, A::Vector{<:Matrix{T}},
        b::Vector{<:Vector{T}}, o::Vector{<:Vector{T}};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in length.(b)]) where T

    np = length(b)
	αr = zeros(T,np)
    αp = zeros(T,np)
    for i = 1:np
        n = length(b[i])
        αr[i], αp[i] = soft_intersection(e, v, ρ, A[i], b[i], o[i], solver=solvers[i])
		# @show αr[i], αp[i]
    end
	values = - (αr + 0.5 * αp ./ ρ)
	α = sum(αr .* softweights(values, -4 - log(ρ) - log(ρ/1e-2)))
    # α = sum(αr .* softweights(values, -log(ρ)))
    return α
end

function soft_intersection(e::Vector, β::Number, ρ, A::Vector{<:Matrix},
		b::Vector{<:Vector}, o::Vector{<:Vector};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in length.(b)])

    v = [cos(β), sin(β)]
    α = soft_intersection(e, v, ρ, A, b, o; solvers=solvers)
    d = e + α * v
    return d
end

function soft_point_cloud(e::Vector, β, ρ, A::Vector{Matrix{T}},
		b::Vector{<:Vector}, o::Vector{<:Vector};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in length.(b)]) where T

    nβ = length(β)
    d = zeros(T, 2, nβ)
    soft_point_cloud!(d, e, β, ρ, A, b, o, solvers=solvers)
    return d
end

function soft_point_cloud!(d::Matrix, e::Vector, β, ρ, A::Vector{<:Matrix},
		b::Vector{<:Vector}, o::Vector{<:Vector};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in length.(b)])

	nβ = length(β)
    for i = 1:nβ
        v = [cos(β[i]), sin(β[i])]
        α = soft_intersection(e, v, ρ, A, b, o, solvers=solvers)
        d[:,i] = e + α * v
    end
    return nothing
end


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
op1 = [0.2, 1.2]

Ap2 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.30ones(4,2)
bp2 = 0.25*[
    +1,
    +1,
    +1,
     1,
    ]
op2 = [-0.8, 1.2]

Af = [0.0  1.0]
bf = [0.0]
of = [0.0, 0.0]
AA = [Ap1, Ap2, Ap, Af]
bb = [bp1, bp2, bp, bf]
oo = [op1, op2, op, of]

op = [0.0, +0.5]
ep = [0.0, +3.0]
vp = [0.0, -1.0]
# vp = 1/sqrt(2) * [-1,-1]
cp = 1e-2
# cp = 1e-1
# cp = 3e-1
parameters = pack_lp_parameters(ep, vp, Ap, bp, op, cp)

build_2d_polytope!(vis[:polytope], Ap, bp + Ap * op, name=:parent)
setobject!(vis[:camera], HyperSphere(MeshCat.Point(ep[1], 0, ep[2]), 0.05))

solver = lp_solver(num_cone, ρ=cp)
set_parameters!(solver, parameters)
solve!(solver)

αr = solver.solution.primals[1]
αp = solver.solution.primals[2]
soft_intersection(ep, vp, cp, Ap, bp, op)
soft_intersection(ep, vp, cp, [Ap, Af], [bp, bf], [op, of])


build_2d_polytope!(vis[:polytope], Ap, αp .+ bp + Ap * op, name=:padded, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope1], Ap1, αp .+ bp1 + Ap1 * op1, name=:padded, color=RGBA(0,0,0,0.3))
build_2d_polytope!(vis[:polytope1], Ap2, αp .+ bp2 + Ap2 * op2, name=:padded2, color=RGBA(0,0,0,0.3))
setobject!(vis[:intersection],
	HyperSphere(MeshCat.Point(0, (ep + αr * vp)...), 0.01))


β = Vector(range(-0.1π, -0.9π, length=100))
d = soft_point_cloud(ep, β, cp, AA, bb, oo)

num_points = size(d, 2)
build_point_cloud!(vis[:point_cloud], num_points; color=RGBA(0.1,0.1,0.1,1), name=Symbol(1))
set_2d_point_cloud!(vis, [ep], [d]; name=:point_cloud)

function soft_loss(d::Vector{<:Matrix}, e::Vector{<:Vector}, β::Vector, ρ, θ::Vector,
		polytope_dimensions::Vector{Int};
		solvers::Vector{<:Solver}=[lp_solver(l, ρ=ρ) for l in polytope_dimensions])

    ne = length(e)
    l = 0.0
	A, b, o = unpack_halfspaces(θ, polytope_dimensions)
    for i = 1:ne
        dθ = soft_point_cloud(e[i], β[i], ρ, A, b, o)
        l += 0.5 * norm(d[i] - dθ)^2 / size(dθ, 2)
    end
    return l
end

θ, polytope_dimensions = pack_halfspaces(AA, bb, oo)
solvers = [lp_solver(l, ρ=cp) for l in polytope_dimensions]
@benchmark soft_loss([d], [ep], [β], cp, θ, polytope_dimensions, solvers=solvers)
