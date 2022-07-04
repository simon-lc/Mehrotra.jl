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


# p0 = rand(35)
# p1 = pack_contact_bundle(unpack_contact_bundle(p0)...)
# norm(p0 - p1)

function mask_contact_bundle(x, θ, indices::Indices228; na::Int=0, nb::Int=0, d::Int=0)

    xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb =
        unpack_lp_simulator_parameters(θ, na=na, nb=nb, d=d)
    y, z, s = x[indices.primals], x[indices.duals], x[indices.slacks]

    # primals
    va25 = y[0 .+ (1:d)]
    ωa25 = y[d .+ (1:1)]
    vb25 = y[d+1 .+ (1:d)]
    ωb25 = y[2d+1 .+ (1:1)]
    xa3 = xa2 + timestep .* va25
    qa3 = qa2 + timestep .* ωa25
    xb3 = xb2 + timestep .* vb25
    qb3 = qb2 + timestep .* ωb25

    subparameters = pack_lp_parameters(xa3, qa3, xb3, qb3, Aa, ba, Ab, bb)
    return subparameters
end

################################################################################
# residual
################################################################################
function composed_residual(x, θ, contact_solver::Solver228, indices::Indices228; na::Int=0, nb::Int=0, d::Int=0)

    subparameters = mask_contact_bundle(x, θ, indices; na=na, nb=nb, d=d)
    # contact bundle
    ϕ, pa, pb, N, ∂pa, ∂pb = contact_bundle(contact_solver, subparameters, na=na, nb=nb, d=d)
    xl = pack_contact_bundle(ϕ, pa, pb, N, ∂pa, ∂pb)

    return composed_residual(x, xl, θ, contact_solver; na=na, nb=nb, d=d)
end

function composed_residual(x, xl, θ, contact_solver::Solver228; na::Int=0, nb::Int=0, d::Int=0)

    xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb =
        unpack_lp_simulator_parameters(θ, na=na, nb=nb, d=d)
    y, z, s = x[indices.primals], x[indices.duals], x[indices.slacks]

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
    ϕ, pa, pb, N, ∂pa, ∂pb = unpack_contact_bundle(xl, d=d)

    # mass matrix
    Ma = Diagonal([mass; mass; inertia])
    Mb = Diagonal([mass; mass; inertia])

    e = [
        Ma * ([xa3; qa3] - 2*[xa2; qa2] + [xa1; qa1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ua * timestep[1];
        Mb * ([xb3; qb3] - 2*[xb2; qb2] + [xb1; qb1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ub * timestep[1];
        sγ - ϕ;
        # sγ .* γ;
    ]
    e[1:2d+2] .+= -N'*γ
    return e
end


function generate_composed_gradients(func::Function, mask::Function,# subfunc::Function,
        num_variables, num_subvariables, num_parameters,
        # dim::Dimensions228, ind::Indices228;
        checkbounds=true,
        threads=false)

    x = Symbolics.variables(:x, 1:num_variables)
    xl = Symbolics.variables(:xl, 1:num_subvariables)
    θ = Symbolics.variables(:θ, 1:num_parameters)

    m = num_parameters > 0 ?
        mask(x, θ) :
        mask(x)

    f = num_parameters > 0 ?
        func(x, xl, θ) :
        func(x, xl)

    mx = Symbolics.sparsejacobian(m, x)
    mθ = Symbolics.sparsejacobian(m, θ)

    fx = Symbolics.sparsejacobian(f, x)
    fxl = Symbolics.sparsejacobian(f, xl)
    fθ = Symbolics.sparsejacobian(f, θ)

    mx_sparsity = collect(zip([findnz(mx)[1:2]...]...))
    mθ_sparsity = collect(zip([findnz(mθ)[1:2]...]...))

    fx_sparsity = collect(zip([findnz(fx)[1:2]...]...))
    fxl_sparsity = collect(zip([findnz(fxl)[1:2]...]...))
    fθ_sparsity = collect(zip([findnz(fθ)[1:2]...]...))


    m_expr = Symbolics.build_function(m, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    mx_expr = Symbolics.build_function(mx.nzval, x, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    mθ_expr = Symbolics.build_function(mθ.nzval, x, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    f_expr = Symbolics.build_function(f, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fx_expr = Symbolics.build_function(fx.nzval, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fxl_expr = Symbolics.build_function(fxl.nzval, x, xl, θ,
        parallel=(threads ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]
    fθ_expr = Symbolics.build_function(fθ.nzval, x, xl, θ,
        parallel=((threads && num_parameters > 0) ? Symbolics.MultithreadedForm() : Symbolics.SerialForm()),
        checkbounds=checkbounds,
        expression=Val{false})[2]

    return m_expr, mx_expr, mθ_expr,
        f_expr, fx_expr, fxl_expr, fθ_expr,
        mx_sparsity, mθ_sparsity,
        fx_sparsity, fxl_sparsity, fθ_sparsity,
        length(m),
        length(f)
end


dimensions = solver.dimensions
x = rand(dimensions.variables)
θ = rand(dimensions.parameters)
composed_residual(x, θ, contact_solver; indices, na=na, nb=nb, d=d)
num_subvariables = 35
sized_composed_residual(x, xl, θ) = composed_residual(x, xl, θ, contact_solver; na=na, nb=nb, d=d)
sized_mask_contact_bundle(x, θ) = mask_contact_bundle(x, θ, indices; na=na, nb=nb, d=d)

m_expr, mx_expr, mθ_expr,
    e_expr, ex_expr, exl_expr, eθ_expr,
    mx_sparsity, mθ_sparsity,
    ex_sparsity, exl_sparsity, eθ_sparsity,
    lm, le = generate_composed_gradients(
        sized_composed_residual, sized_mask_contact_bundle,
        dimensions.variables,
        num_subvariables,
        dimensions.parameters)


function e_test(e, x, xl, θ, θl, m_expr, e_expr, subfunc::Function)
    # m_expr(θl, x, θ)
    subfunc(xl, θl)
    # e_expr(e, x, xl, θ)
    return nothing
end

num_subparameters = lm
num_variables = solver.dimensions.variables
num_equality = solver.dimensions.equality
θl = zeros(num_subparameters)
θ = rand(num_parameters)
xl = zeros(num_subvariables)
x = rand(num_variables)
e = rand(num_equality)
eee(x, xl, θ, θl, m_expr, e_expr)

contact_solver.options.verbose = false
subfunc(subvariables, subparameters) = contact_bundle(subvariables, subparameters, contact_solver; na=na, nb=nb, d=d)

Main.@code_warntype e_test(e, x, xl, θ, θl, m_expr, e_expr, subfunc)
@benchmark $e_test($e, $x, $xl, $θ, $θl, $m_expr, $e_expr, $subfunc)






m_expr

e, ex, eθ, ex_sparsity, eθ_sparsity = generate_gradients(equality, dim, idx)

methods = ProblemMethods228(
    e,
    ex,
    eθ,
    zeros(length(ex_sparsity)),
    zeros(length(eθ_sparsity)),
    ex_sparsity,
    eθ_sparsity,
    # c, cx, cθ,
    #     zeros(length(cx_sparsity)), zeros(length(cθ_sparsity)),
    #     cx_sparsity, cθ_sparsity,
)



















"""
    v = variables [xa, qa, xb, qb]
    ϕ = signed distance function
    N = ∂ϕ∂v jacobian
    pa = contact point on body a in world coordinates
    pb = contact point on body b in world coordinates
    ∂pa = ∂pa∂v jacobian, derivative of the contact point location not attached to body a
    ∂pb = ∂pb∂v jacobian, derivative of the contact point location not attached to body a
"""
struct ContactBundle111{T}
    name::Symbol
    parent_id::Int
    child_id::Int
    ϕ::Vector{T}
    N::Matrix{T}
    pa::Vector{T}
    pb::Vector{T}
    ∂pa::Matrix{T}
    ∂pb::Matrix{T}
end



function lp_simulator_residual_jacobian_variables(primals, duals, slacks, parameters, contact_solver;
        na::Int=0, nb::Int=0, d::Int=0)

    FiniteDiff.finite_difference_jacobian(
        x -> lp_simulator_residual(x[indices.primals], x[indices.duals], x[indices.slacks], parameters, contact_solver;
            na=na, nb=nb, d=d),
        [primals; duals; slacks])
end

function lp_simulator_residual_jacobian_parameters(primals, duals, slacks, parameters, contact_solver;
        na::Int=0, nb::Int=0, d::Int=0)

    FiniteDiff.finite_difference_jacobian(
        parameters -> lp_simulator_residual(primals, duals, slacks, parameters, contact_solver;
            na=na, nb=nb, d=d),
        parameters)
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

function extract_contact_parameters(primals, duals, slacks, parameters; na::Int=0, nb::Int=0, d::Int=0)

    xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb =
        unpack_lp_simulator_parameters(parameters, na=na, nb=nb, d=d)

    va25 = primals[0 .+ (1:d)]
    ωa25 = primals[d .+ (1:1)]
    vb25 = primals[d+1 .+ (1:d)]
    ωb25 = primals[d+2 .+ (1:1)]
    xa3 = xa2 + timestep .* va25
    qa3 = qa2 + timestep .* ωa25
    xb3 = xb2 + timestep .* vb25
    qb3 = qb2 + timestep .* ωb25

    contact_parameters = pack_lp_parameters(xa3, qa3, xb3, qb3, Aa, ba, Ab, bb, na=na, nb=nb, d=d)
    return contact_parameters
end


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
        @show i, H
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

        d = 2
        va15 .= solver.solution.primals[0 .+ (1:d)]
        ωa15 .= solver.solution.primals[d .+ (1:1)]
        vb15 .= solver.solution.primals[d+1 .+ (1:d)]
        ωb15 .= solver.solution.primals[2d+1 .+ (1:1)]
        xa2 = xa2 + timestep * va15
        qa2 = qa2 + timestep * ωa15
        xb2 = xb2 + timestep * vb15
        qb2 = qb2 + timestep * ωb15

        # pw = solver.solution.primals[1:d]
        # ϕ = solver.solution.primals[d+1]
        γ = solver.solution.duals[1:1]
        sγ = solver.solution.slacks[1:1]
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
d = 2

contact_solver = lp_contact_solver(Aa, ba, Ab, bb; d=d, options=Options228(verbose=false))

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
inertia = 0.4
gravity = -0.0*9.81

parameters = pack_lp_simulator_parameters(
    xa2, qa2, xb2, qb2,
    va15, ωa15, vb15, ωb15,
    u, timestep, mass, inertia, gravity,
    Aa, ba, Ab, bb)

num_primals = 2d+2
num_cone = 1
num_parameters = length(parameters)
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

dimensions = Dimensions(num_primals, num_cone, num_parameters)
indices = Indices(num_primals, num_cone, num_parameters)


sized_lp_simulator_residual(primals, duals, slacks, parameters) =
    lp_simulator_residual(primals, duals, slacks, parameters, contact_solver; na=na, nb=nb, d=d)
sized_lp_simulator_residual_jacobian_variables(primals, duals, slacks, parameters) =
    lp_simulator_residual_jacobian_variables(primals, duals, slacks, parameters, contact_solver; na=na, nb=nb, d=d)
sized_lp_simulator_residual_jacobian_parameters(primals, duals, slacks, parameters) =
    lp_simulator_residual_jacobian_parameters(primals, duals, slacks, parameters, contact_solver; na=na, nb=nb, d=d)


lp_simulator_methods = generate_problem_methods(
    sized_lp_simulator_residual,
    sized_lp_simulator_residual_jacobian_variables,
    sized_lp_simulator_residual_jacobian_parameters,
    dimensions,
    indices)


solver = Solver(nothing, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    methods=lp_simulator_methods,
    options=Options228(max_iterations=30, verbose=true)
    )

solve!(solver)

################################################################################
# simulation
################################################################################
xa2 = [2.0,2.0]
xb2 = [0.0,2.0]
qa2 = [+0.5]
qb2 = [-0.5]

va15 = [-1.0,0.0]
ωa15 = [+3.0]
vb15 = [0,0.0]
ωb15 = [+0.0]
H = 300
U = [zeros(2d+2) for i=1:H]

Pw, Xa, Qa, Xb, Qb, Va, Ωa, Vb, Ωb, iterations = simulate_lp(
    solver, xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U;
    timestep=timestep,
    mass=mass,
    inertia=inertia,
    friction_coefficient=0.2,
    gravity=gravity)


################################################################################
# visualization
################################################################################
build_2d_polyhedron!(vis, Aa, ba, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polyhedron!(vis, Ab, bb, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_2d_polyhedron!(vis, Xa[i], Qa[i], name=:polya)
        set_2d_polyhedron!(vis, Xb[i], Qb[i], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Pw[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)



# open(vis)
# convert_frames_to_video_and_gif("polytope_no_gravity")
