using Plots
using MeshCat
using Polyhedra
using GeometryBasics
using RobotVisualizer
using Quaternions
using StaticArrays

include("../polyhedron.jl")
include("../visuals.jl")
include("../quaternion.jl")
include("../rotate.jl")
include("../contact_model/lp_2d.jl")

################################################################################
# residual
################################################################################
function lp_simulator_residual(primals, duals, slacks, parameters, contact_solver::Solver228; na::Int=0, nb::Int=0, d::Int=0)

    xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb =
        unpack_lp_simulator_parameters(parameters, na=na, nb=nb, d=d)

    y, z, s = primals, duals, slacks

    # duals
    γ = z[1:1]
    # slacks
    sγ = s[1:1]

    # primals
    # velocity
    va25 = y[0 .+ (1:d)]
    ωa25 = y[d .+ (1:1)]
    vb25 = y[d+1 .+ (1:d)]
    ωb25 = y[2d+1 .+ (1:1)]
    xa1 = xa2 - timestep .* va15
    qa1 = qa2 - timestep .* ωa15
    xb1 = xb2 - timestep .* vb15
    qb1 = qb2 - timestep .* ωb15
    xa3 = xa2 + timestep .* va25
    qa3 = qa2 + timestep .* ωa25
    xb3 = xb2 + timestep .* vb25
    qb3 = qb2 + timestep .* ωb25

    # controls
    ua = u[1:d+1]
    ub = u[d+1 .+ (1:d+1)]

    # contact bundle
    ϕ, pa, pb, N, ∂pa, ∂pb = contact_bundle(contact_solver, xa3, qa3, xb3, qb3, na=na, nb=nb, d=d)

    # mass matrix
    Ma = Diagonal([mass; mass; inertia])
    Mb = Diagonal([mass; mass; inertia])

    res = [
        Ma * ([xa3; qa3] - 2*[xa2; qa2] + [xa1; qa1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ua * timestep[1];# - Na'*γ;
        Mb * ([xb3; qb3] - 2*[xb2; qb2] + [xb1; qb1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ub * timestep[1];# - Nb'*γ;
        sγ - ϕ;
        # sγ .* γ;
    ]
    res[1:2d+2] .+= -N'*γ
    return res
end

function lp_simulator_residual_jacobian_variables(primals, duals, slacks, parameters, contact_solver; na::Int=0, nb::Int=0, d::Int=0)

    FiniteDiff.finite_difference_jacobian(
        x -> lp_simulator_residual(x[indices.primals], x[indices.duals], x[indices.slacks], parameters, contact_solver;
            na=na, nb=nb, d=d),
        [primals; duals; slacks])

    return nothing
end

function lp_simulator_residual_jacobian_parameters(primals, duals, slacks, parameters, contact_solver; na::Int=0, nb::Int=0, d::Int=0)

    FiniteDiff.finite_difference_jacobian(
        parameters -> lp_simulator_residual(primals, duals, slacks, parameters, contact_solver;
            na=na, nb=nb, d=d),
        parameters)

    return nothing
end

function generate_problem_methods(equality_constraint, equality_jacobian_variables,
        equality_jacobian_parameters, dimensions::Dimensions228, indices::Indices228)

    function e(v, x, θ)
        primals = x[indices.primals]
        duals = x[indices.duals]
        slacks = x[indices.slacks]
        parameters = θ
        v .= equality_constraint(primals, duals, slacks, parameters)
        return nothing
    end

    function ex(v, x, θ)
        primals = x[indices.primals]
        duals = x[indices.duals]
        slacks = x[indices.slacks]
        parameters = θ
        v .= vec(equality_jacobian_variables(primals, duals, slacks, parameters))
        return nothing
    end

    function eθ(v, x, θ)
        primals = x[indices.primals]
        duals = x[indices.duals]
        slacks = x[indices.slacks]
        parameters = θ
        v .= vec(equality_jacobian_parameters(primals, duals, slacks, parameters))
        return nothing
    end

    ex_sparsity = collect(zip([findnz(ones(dimensions.equality, dimensions.variables))[1:2]...]...))
    eθ_sparsity = collect(zip([findnz(ones(dimensions.equality, dimensions.parameters))[1:2]...]...))

    methods = ProblemMethods228(
        e,
        ex,
        eθ,
        zeros(length(ex_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        eθ_sparsity,
    )
    return methods
end

dimensions = Dimensions(num_primals, num_cone, num_parameters)
indices = Indices(num_primals, num_cone, num_parameters)
generate_problem_methods(
    lp_simulator_residual,
    lp_simulator_residual_jacobian_variables,
    lp_simulator_residual_jacobian_parameters,
    dimensions,
    indices)


function dummy_residual(primals, duals, slacks, parameters)
    res = zeros(length(primals) + lengh(duals))
    return res
end

solver = Solver(dummy_residual, num_primals, num_cone, num_parameters)


# function extract_contact_parameters(primals, duals, slacks, parameters; na::Int=0, nb::Int=0, d::Int=0)
#
#     xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb =
#         unpack_lp_simulator_parameters(parameters, na=na, nb=nb, d=d)
#
#     va25 = primals[0 .+ (1:d)]
#     ωa25 = primals[d .+ (1:1)]
#     vb25 = primals[d+1 .+ (1:d)]
#     ωb25 = primals[d+2 .+ (1:1)]
#     xa3 = xa2 + timestep .* va25
#     qa3 = qa2 + timestep .* ωa25
#     xb3 = xb2 + timestep .* vb25
#     qb3 = qb2 + timestep .* ωb25
#
#     contact_parameters = pack_lp_parameters(xa3, qa3, xb3, qb3, Aa, ba, Ab, bb, na=na, nb=nb, d=d)
#     return contact_parameters
# end


################################################################################
# parameters
################################################################################
function unpack_lp_simulator_parameters(parameters; na=0, nb=0, d=0)
    off = 0
    xa = parameters[off .+ (1:d)]; off += d
    qa = parameters[off .+ (1:1)]; off += 1
    xb = parameters[off .+ (1:d)]; off += d
    qb = parameters[off .+ (1:1)]; off += 1

    va15 = parameters[off .+ (1:d)]; off += d
    ωa15 = parameters[off .+ (1:1)]; off += 1
    vb15 = parameters[off .+ (1:d)]; off += d
    ωb15 = parameters[off .+ (1:1)]; off += 1

    u = parameters[off .+ (1:2d+2)]; off += 2d+2
    timestep = parameters[off .+ (1:1)]; off += 1
    mass = parameters[off .+ (1:1)]; off += 1
    inertia = parameters[off .+ (1:1)]; off += 1
    gravity = parameters[off .+ (1:1)]; off += 1

    Aa = parameters[off .+ (1:na*d)]; off += na*d
    Aa = reshape(Aa, (na,d))
    ba = parameters[off .+ (1:na)]; off += na

    Ab = parameters[off .+ (1:nb*d)]; off += nb*d
    Ab = reshape(Ab, (nb,d))
    bb = parameters[off .+ (1:nb)]; off += nb

    return xa, qa, xb, qb, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb
end

function pack_lp_simulator_parameters(xa, qa, xb, qb, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb)
    return [xa; qa; xb; qb; va15; ωa15; vb15; ωb15; u; timestep; mass; inertia; gravity; vec(Aa); ba; vec(Ab); bb]
end

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

function simulate_lp(solver, xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U;
        timestep=0.01,
        mass=1.0,
        inertia=0.1,
        friction_coefficient=0.2,
        gravity=-9.81)

    H = length(U)
    Pw = []
    Xa = []
    Qa = []
    Xb = []
    Qb = []
    Va = []
    Vb = []
    Ωa = []
    Ωb = []
    iterations = Vector{Int}()

    pw = xa2 # fake

    for i = 1:H
        push!(Pw, pw)
        push!(Xa, xa2)
        push!(Qa, qa2)
        push!(Xb, xb2)
        push!(Qb, qb2)
        push!(Va, va15)
        push!(Ωa, ωa15)
        push!(Vb, vb15)
        push!(Ωb, ωb15)
        parameters = pack_lp_simulator_parameters(xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U[i], timestep, mass, inertia, gravity, Aa, ba, Ab, bb)
        solver.parameters .= parameters

        solver.options.verbose = false
        solve!(solver)
        push!(iterations, solver.trace.iterations)

        va15 .= solver.solution.primals[d+1 .+ (1:2)]
        ωa15 .= solver.solution.primals[2d+1 .+ (1:1)]
        vb15 .= solver.solution.primals[2d+2 .+ (1:2)]
        ωb15 .= solver.solution.primals[3d+2 .+ (1:1)]
        xa2 = xa2 + timestep * va15
        qa2 = qa2 + timestep * ωa15
        xb2 = xb2 + timestep * vb15
        qb2 = qb2 + timestep * ωb15

        pw = solver.solution.primals[1:d]
        ϕ = solver.solution.primals[d+1]
        γ = solver.solution.duals[na+nb+1]
        sγ = solver.solution.slacks[na+nb+1]
        @show ϕ
        @show sγ
        @show γ
        pa = x_2d_rotation(qa2)' * (pw - xa2)
        pb = x_2d_rotation(qb2)' * (pw - xb2)
    end
    return Pw, Xa, Qa, Xb, Qb, Va, Ωa, Vb, Ωb, iterations
end















################################################################################
# demo
################################################################################
vis = Visualizer()
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

Aa = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
ba = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]

Ab = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bb = 0.5*[
     1,
     1,
     1,
     1,
    ]
na = length(ba)
nb = length(bb)

contact_solver = lp_contact_solver(Aa, ba, Ab, bb; d=2)

xa2 = [1,3.0]
xb2 = [0,4.0]
qa2 = [+0.5]
qb2 = [-0.5]

va15 = [0,0.0]
ωa15 = [+1.0]
vb15 = [0,0.0]
ωb15 = [-1.0]

u = zeros(2d+2)
timestep = 0.01
mass = 1.0
inertia = 0.1
gravity = -0.0*9.81

parameters = pack_lp_simulator_parameters(
    xa2, qa2, xb2, qb2,
    va15, ωa15, vb15, ωb15,
    u, timestep, mass, inertia, gravity,
    Aa, ba, Ab, bb)




d = 2
num_primals = 2d+2
num_cone = 1
num_parameters = length(parameters)

sized_lp_simulator_residual(primals, duals, slacks, parameters) =
    lp_simulator_residual(primals, duals, slacks, parameters, contact_solver; na=na, nb=nb, d=d)
indices = Indices(num_primals, num_cone, num_parameters)
generate_problem_methods(sized_lp_simulator_residual, nothing, nothing, indices)


primals = rand(num_primals)
duals = rand(num_cone)
slacks = rand(num_cone)
parameters = rand(num_parameters)
lp_simulator_residual(primals, duals, slacks, parameters, contact_solver;
    na=na, nb=nb, d=d)
lp_simulator_residual_jacobian_variables(primals, duals, slacks, parameters, contact_solver;
    na=na, nb=nb, d=d)
lp_simulator_residual_jacobian_parameters(primals, duals, slacks, parameters, contact_solver;
    na=na, nb=nb, d=d)
