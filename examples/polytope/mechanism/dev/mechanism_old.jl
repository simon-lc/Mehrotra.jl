
# function step!(mechanism::Mechanism171{T}, x::Vector{T}, u::Vector{T}) where T
# end
#
# function input_gradient(du, x, u, mechanism::Mechanism171{T})
# end
#
# function state_gradient(dx, x, u, mechanism::Mechanism171{T})
# end
#
# function set_input!(mechanism::Mechanism171{T})
# end
#
# function set_current_state!(mechanism::Mechanism171{T})
# end
#
# function set_next_state!(mechanism::Mechanism171{T})
# end
#
# function get_current_state!(mechanism::Mechanism171{T})
# end
#
# function get_next_state!(mechanism::Mechanism171{T})
# end


using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions


vis = Visualizer()
render(vis)

include("../contact_model/lp_2d.jl")
include("../polytope.jl")
include("../visuals.jl")
include("../rotate.jl")
include("../quaternion.jl")

include("node.jl")
include("body.jl")
include("contact.jl")





################################################################################
# dimensions
################################################################################
struct MechanismDimensions171
    body_configuration::Int
    body_velocity::Int
    body_state::Int
    bodies::Int
    contacts::Int
    variables::Int
    parameters::Int
    primals::Int
    cone::Int
    equality::Int
end

function MechanismDimensions171(bodies::Vector, contacts::Vector)
    # dimensions
    nq = 3 # in 2D
    nv = 3 # in 2D
    nx = 6 # in 2D
    nb = length(bodies)
    nc = length(contacts)
    nx = sum(variable_dimension.(bodies)) + sum(variable_dimension.(contacts))
    nθ = sum(parameter_dimension.(bodies)) + sum(parameter_dimension.(contacts))
    num_primals = sum(variable_dimension.(bodies))
    num_cone = Int(sum(variable_dimension.(contacts)) / 2)
    num_equality = num_primals + num_cone
    return MechanismDimensions171(nq, nv, nx, nb, nc, nx, nθ, num_primals, num_cone, num_equality)
end

################################################################################
# mechanism
################################################################################
struct Mechanism171{T,D,NB,NC}
    variables::Vector{T}
    parameters::Vector{T}
    solver::Solver{T}
    bodies::Vector{Body171{T}}
    contacts::Vector{Contact171{T}}
    dimensions::MechanismDimensions171
    # equalities::Vector{Equality{T}}
    # inequalities::Vector{Inequality{T}}
end

function Mechanism171(residual, bodies::Vector, contacts::Vector;
        options::Options{T}=Options(), D::Int=2) where {T}

    # Dimensions
    dim = MechanismDimensions171(bodies, contacts)

    # indexing
    indexing!([bodies; contacts])

    # solver
    parameters = vcat(get_parameters.(bodies)..., get_parameters.(contacts)...)

    # methods = mechanism_methods(bodies, contacts, dim)
    solver = Solver(
            residual,
            dim.primals,
            dim.cone,
            parameters=parameters,
            nonnegative_indices=collect(1:dim.cone),
            second_order_indices=[collect(1:0)],
            method_type=:finite_difference,
            options=options
            )
    # solver = Solver(
    #         nothing,
    #         dim.primals,
    #         dim.cone,
    #         parameters=parameters,
    #         nonnegative_indices=collect(1:dim.cone),
    #         second_order_indices=[collect(1:0)],
    #         methods=methods,
    #         options=options
    #         )

    # vectors
    variables = solver.solution.all
    parameters = solver.parameters

    nb = length(bodies)
    nc = length(contacts)
    mechanism = Mechanism171{T,D,nb,nc}(
        variables,
        parameters,
        solver,
        bodies,
        contacts,
        dim,
        )
    return mechanism
end

function indexing!(nodes::Vector)
    eoff = 0
    xoff = 0
    θoff = 0
    for node in nodes
        ne = equality_dimension(node)
        nx = variable_dimension(node)
        nθ = parameter_dimension(node)
        node.node_index.e = collect(eoff .+ (1:ne)); eoff += ne
        node.node_index.x = collect(xoff .+ (1:nx)); xoff += nx
        node.node_index.θ = collect(θoff .+ (1:nθ)); θoff += nθ
    end
    return nothing
end


################################################################################
# demo
################################################################################
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
bp = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bc = 0.5*[
     1,
     1,
     1,
     1,
    ]


timestep = 0.01
gravity = -0.0*9.81
mass = 1.0
inertia = 0.2 * ones(1,1)
bodya = Body171(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body171(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact171(bodies[1], bodies[2])]

dim = MechanismDimensions171(bodies, contacts)
indexing!([bodies; contacts])

x0 = rand(dim.variables)
θ0 = rand(dim.parameters)
e0 = zeros(dim.variables)
ex0 = zeros(dim.variables, dim.variables)
eθ0 = zeros(dim.variables, dim.parameters)


function mechanism_residual(primals, duals, slacks, parameters; dim::MechanismDimensions171)
    e = zeros(dim.equality)
    x = [primals; duals; slacks]
    θ = parameters

    # body
    for body in bodies
        body_residual!(e, x, θ, body)
    end

    # contact
    for contact in contacts
        contact_residual!(e, x, θ, contact, bodies[1], bodies[2])
    end
    return e
end

local_residual(primals, duals, slacks, parameters) =
    mechanism_residual(primals, duals, slacks, parameters, dim)
mech = Mechanism171(local_residual, bodies, contacts)
solver = mech.solver
solve!(solver)











contact_solver = ContactSolver(Ap, bp, Ac, bc)
contact_methods = ContactMethods171(contacts[1], bodies..., dim)

problem_methods0 = mechanism_methods(bodies, contacts, dim)
methods0 = problem_methods0.methods
# evaluate!(e0, ex0, eθ0, x0, θ0, methods0)
# Main.@profiler [evaluate!(e0, ex0, eθ0, x0, θ0, methods0) for i=1:5000]
# @benchmark $evaluate!($e0, $ex0, $eθ0, $x0, $θ0, $methods0)




dim_solver = Dimensions(dim.primals, dim.cone, dim.parameters)
index_solver = Indices(dim.primals, dim.cone, dim.parameters)
problem = ProblemData(dim.variables, dim.parameters, dim.equality, dim.cone)
idx_nn = collect(1:dim.cone)
idx_soc = [collect(1:0)]
cone_methods = ConeMethods(dim.cone, idx_nn, idx_soc)
solution = Point(dim_solver, index_solver)
solution.all .= 1.0
parameters = ones(dim.parameters)

body_method0 = methods0[1]
contact_method0 = methods0[3]

evaluate!(
        problem,
        problem_methods0,
        cone_methods,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        )
Main.@code_warntype evaluate!(
        problem,
        problem_methods0,
        cone_methods,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        )
@benchmark $evaluate!(
        $problem,
        $problem_methods0,
        $cone_methods,
        $solution,
        $parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true,
        cone_constraint=true,
        cone_jacobian=true,
        cone_jacobian_inverse=true,
        )

evaluate!(problem,
        contact_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
Main.@code_warntype evaluate!(problem,
        contact_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
@benchmark $evaluate!($problem,
        $contact_method0,
        $solution,
        $parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)

evaluate!(problem,
        body_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
Main.@code_warntype evaluate!(problem,
        body_method0,
        solution,
        parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)
@benchmark $evaluate!($problem,
        $body_method0,
        $solution,
        $parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
        equality_jacobian_parameters=true)












################################################################################
# test mechanism
################################################################################
mech = Mechanism171(bodies, contacts)
solver = mech.solver

bodies[1].pose .= [+10,0,0.0]
bodies[2].pose .= [-10,0,0.0]

θb1 = get_parameters(bodies[1])
θb2 = get_parameters(bodies[2])
θc1 = get_parameters(contacts[1])
parameters = [θb1; θb2; θc1]

solver.parameters .= parameters
evaluate!(solver.problem,
    solver.methods,
    solver.cone_methods,
    solver.solution,
    solver.parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    equality_jacobian_parameters=true,
    cone_constraint=true,
    cone_jacobian=true,
    cone_jacobian_inverse=true,
    )

function residual(variables, parameters, bodies, contacts, contact_methods)
    num_equality = 7
    num_variables = length(variables)

    x = variables
    θ = parameters

    e = zeros(num_equality)
    for body in bodies
        body_residual!(e, x, θ, body)
    end
    for contact in contacts
        # update xl = [ϕ, pa, pb, N, ∂pa, ∂pb]
        contact_solver = contact_methods[1].contact_solver
        xl = contact_methods[1].subvariables
        θl = contact_methods[1].subparameters
        contact_methods[1].set_subparameters!(θl, x, θ)
        update_subvariables!(xl, θl, contact_solver)
        contact_residual!(e, x, xl, θ, contact, bodies[1], bodies[2])
    end
    return e
end


solver.solution.all .= [+0,0,0, -0,0,0, 1e-7, 10.5]
evaluate!(solver.problem,
    solver.methods,
    solver.cone_methods,
    solver.solution,
    solver.parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    equality_jacobian_parameters=true,
    cone_constraint=true,
    cone_jacobian=true,
    cone_jacobian_inverse=true,
    )

e0 = residual(solver.solution.all, solver.parameters, bodies, contacts, methods0[3:3])
e1 = deepcopy(solver.problem.equality_constraint)
e0
e1
norm(e0 - e1, Inf)

J0 = FiniteDiff.finite_difference_jacobian(
    variables -> residual(variables, solver.parameters, mech.bodies, mech.contacts,
        methods0[3:3]), mech.solver.solution.all)
J1 = deepcopy(mech.solver.problem.equality_jacobian_variables)
norm(J0 - J1, Inf)

mech.bodies[1].pose
mech.bodies[2].pose


using Plots
plot(Gray.(1e3abs.(J0)))
plot(Gray.(1e3abs.(J1)))
plot(Gray.(1e3abs.(J1 - J0)))

J0[3,:]
J1[3,:]
J0[6,:]
J1[6,:]


# function finite_diff_jacobian(solver)
#     evaluate!(solver.problem,
#         solver.methods,
#         solver.cone_methods,
#         solver.solution,
#         solver.parameters,
#         equality_constraint=true,
#         equality_jacobian_variables=true,
#         equality_jacobian_parameters=true,
#         cone_constraint=true,
#         cone_jacobian=true,
#         cone_jacobian_inverse=true,
#         )
#
#
#     return

parameter_dimension(bodies[1])
parameter_dimension(contacts[1])
mech.solver.options.verbose = false
@benchmark $solve!($mech.solver)
Main.@profiler [solve!(mech.solver) for i=1:1000]





################################################################################
# test simulation
################################################################################

mech = Mechanism171(bodies, contacts)
mech.solver.methods.methods[3].contact_solver.solver.options.complementarity_tolerance=1e-2
mech.solver.methods.methods[3].contact_solver.solver.options.residual_tolerance=1e-10
# mech.solver.methods.methods[3].contact_solver.solver.options.verbose=true
Xa2 = [[+1,1,0.0]]
Xb2 = [[-1,1,0.0]]
Va15 = [[-2,0,0.0]]
Vb15 = [[+0,0,0.0]]
Pp = []
Pc = []
iter = []

H = 200
for i = 1:H
    mech.bodies[1].pose .= Xa2[end]
    mech.bodies[1].velocity .= Va15[end]
    mech.bodies[2].pose .= Xb2[end]
    mech.bodies[2].velocity .= Vb15[end]

    θb1 = get_parameters(mech.bodies[1])
    θb2 = get_parameters(mech.bodies[2])
    θc1 = get_parameters(mech.contacts[1])
    mech.parameters .= [θb1; θb2; θc1]
    mech.solver.parameters .= [θb1; θb2; θc1]

    # mech = Mechanism171(bodies, contacts)
    solve!(mech.solver)
    va25 = deepcopy(mech.solver.solution.all[1:3])
    vb25 = deepcopy(mech.solver.solution.all[4:6])
    push!(Va15, va25)
    push!(Vb15, vb25)
    push!(Xa2, Xa2[end] + timestep * va25)
    push!(Xb2, Xb2[end] + timestep * vb25)
    p_parent = deepcopy(mech.solver.methods.methods[3].subvariables[2:3]) + (Xa2[end][1:2] + Xb2[end][1:2])/2
    p_child = deepcopy(mech.solver.methods.methods[3].subvariables[4:5]) + (Xa2[end][1:2] + Xb2[end][1:2])/2
    push!(Pp, p_parent)
    push!(Pc, p_child)
    push!(iter, mech.solver.trace.iterations)
end

using Plots
scatter(iter)

################################################################################
# visualization
################################################################################
set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polytope!(vis, Ap, bp, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ac, bc, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_2d_polytope!(vis, Xa2[i][1:2], Xa2[i][3:3], name=:polya)
        set_2d_polytope!(vis, Xb2[i][1:2], Xb2[i][3:3], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Pp[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
render(vis)
# open(vis)
# convert_frames_to_video_and_gif("polytope_thrashing_collision")

# n = 2
# m = 5
# @variables a[1:n]
# @variables r[1:m]
# a = Symbolics.scalarize(a)
# r = Symbolics.scalarize(r)
#
# out = 1.0*r
# out[2 .+ (1:n)] .+= a
# r
# expr = build_function(out, r, a)[2]
# ftest = eval(expr)
#
# out0 = zeros(m)
# r0 = zeros(m)
# a0 = ones(n)
# ftest(out0, r0, a0)
# out0
# @benchmark $ftest($out0, $r0, $a0)


#
# num_variables = dim.variables
# num_parameters = dim.parameters
# @variables out[1:num_variables]
# @variables r[1:num_variables]
# @variables x[1:num_variables]
# @variables θ[1:num_parameters]
# out = Symbolics.scalarize(out)
# r = Symbolics.scalarize(r)
# x = Symbolics.scalarize(x)
# θ = Symbolics.scalarize(θ)
#
# indexing!([bodies; contacts])
# out .= r
# residuals[1](out, x, θ)
# r
# out
# symbolic_residual = eval(build_function(out, r, x, θ)[2])
#
# r0 = ones(num_variables)
# x0 = zeros(num_variables)
# θ0 = ones(num_parameters)
# symbolic_residual(r0, r0, x0, θ0)
# r0
# @benchmark $symbolic_residual($r0, $r0, $x0, $θ0)
#
#
# generate_residual(residuals[1], num_variables, num_parameters)
# generate_residual(residuals[2], num_variables, num_parameters)
