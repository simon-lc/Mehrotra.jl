

function body_residual!(e, x, θ, body::Body177)
    index = body.index
    # variables = primals = velocity
    v25 = unpack_body_variables(x[index.x])
    # parameters
    p2, v15, u, timestep, gravity, mass, inertia = unpack_body_parameters(θ[index.θ], body)
    # integrator
    p1 = p2 - timestep[1] * v15
    p3 = p2 + timestep[1] * v25
    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])
    # dynamics
    dynamics = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    e[index.e] .+= dynamics
    return nothing
end


function contact_residual!(e, x, xl2, xl3, θ, contact::Contact177, pbody::Body177, cbody::Body177)
    # variables
    γ, sγ = unpack_contact_variables(x[contact.index.x])
    # subvariables
    _, _, _, N2, _, _ = unpack_contact_subvariables(xl2, contact)
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl3, contact)
    # @show ϕ
    # dynamics
    e[contact.index.e] .+= sγ - (ϕ .- 0.0)
    e[[pbody.index.e; cbody.index.e]] .+= -N2'*γ
    return nothing
end

function equality_residual(x, θ, bodies, contacts, contact_solver)
    e = zeros(num_equality)
    # @show θ[1:3]
    for body in bodies
        body_residual!(e, x, θ, body)
    end
    for contact in contacts
        num_subvariables = 131
        xp2 = unpack_body_parameters(θ[bodies[1].index.θ], bodies[1])[1]
        xc2 = unpack_body_parameters(θ[bodies[2].index.θ], bodies[2])[1]
        timestep = unpack_body_parameters(θ[bodies[2].index.θ], bodies[2])[4]

        vp25 = unpack_body_variables(x[bodies[1].index.x])
        vc25 = unpack_body_variables(x[bodies[2].index.x])
        xp3 = xp2 + timestep[1] * vp25
        xc3 = xc2 + timestep[1] * vc25

        xl2 = zeros(num_subvariables)
        # @show xp3, xc3
        contact_solver.solver.parameters .= pack_lp_parameters(xp2[1:2], xp2[3:3], xc2[1:2], xc2[3:3], Ap, bp, Ac, bc)
        solve!(contact_solver.solver)
        extract_subvariables!(xl2, contact_solver.solver)

        xl3 = zeros(num_subvariables)
        contact_solver.solver.parameters .= pack_lp_parameters(xp3[1:2], xp3[3:3], xc3[1:2], xc3[3:3], Ap, bp, Ac, bc)
        solve!(contact_solver.solver)
        extract_subvariables!(xl3, contact_solver.solver)

        contact_residual!(e, x, xl2, xl3, θ, contact, bodies[1], bodies[2])
    end
    return deepcopy(e)
end


################################################################################
# demo
################################################################################
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
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.00ones(4,2)
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
bodya = Body177(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body177(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact177(bodies[1], bodies[2])]
indexing!([bodies; contacts])
contact_solver = ContactSolver(Ap, bp, Ac, bc,
    options=Options(
    verbose=false,
    complementarity_tolerance=1e-4,
    residual_tolerance=1e-6,
    ))


num_equality = sum(equality_dimension.(bodies)) + sum(equality_dimension.(contacts))
num_variables = sum(variable_dimension.(bodies)) + sum(variable_dimension.(contacts))
num_parameters = sum(parameter_dimension.(bodies)) + sum(parameter_dimension.(contacts))
num_primals = 3 + 3
num_cone = 1


function evaluate!(
        problem::ProblemData{T},
        methods,
        cone_methods::ConeMethods{T,B,BX,P,PX,PXI,TA},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        cone_jacobian_inverse=false,
        ) where {T,B,BX,P,PX,PXI,TA}

    # reset
    problem.equality_constraint .= 0.0
    problem.equality_jacobian_variables .= 0.0
    problem.equality_jacobian_parameters .= 0.0

    if equality_constraint
        # @show size(methods.e(x, θ))
        # @show size(problem.equality_constraint)
        problem.equality_constraint .= methods.e(solution.all, parameters)
    end

    if equality_jacobian_variables
        # @show size(methods.ex(x, θ))
        # @show size(problem.equality_jacobian_variables)
        problem.equality_jacobian_variables .= methods.ex(solution.all, parameters)
    end

    if equality_jacobian_parameters
        problem.equality_jacobian_parameters .= methods.eθ(solution.all, parameters)
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
        cone_jacobian_inverse=cone_jacobian_inverse,
        cone_target=true # TODO this should only be true once at the beginning of the solve
    )

    return nothing
end

struct FiniteDiffMethods113{T,E,EX,Eθ} <: AbstractProblemMethods{T,E,EX,EP}
    e::E
    ex::EX
    eθ::Eθ
    α::T
end

e_expr(x, θ) = equality_residual(x, θ, bodies, contacts, contact_solver)
function ex_expr(x, θ)
    J = FiniteDiff.finite_difference_jacobian(
        x -> equality_residual(x, θ, bodies, contacts, contact_solver), x)
    return J
end
function eθ_expr(x, θ)
    J = FiniteDiff.finite_difference_jacobian(
        θ -> equality_residual(x, θ, bodies, contacts, contact_solver), θ)
    return J
end

finite_diff_methods = FiniteDiffMethods113(e_expr, ex_expr, eθ_expr, 1.0)


solver = Solver(
        nothing,
        num_primals,
        num_cone,
        parameters=zeros(num_parameters),
        nonnegative_indices=collect(1:num_cone),
        second_order_indices=[collect(1:0)],
        methods=finite_diff_methods,
        options=Options()
        )




bodies[1].pose .= [+0.40,0,0.0]
bodies[2].pose .= [-0.40,0,0.0]

bodies[1].velocity .= [+0,0,0.0]
bodies[2].velocity .= [-0,0,0.0]

θb1 = get_parameters(bodies[1])
θb2 = get_parameters(bodies[2])
θc1 = get_parameters(contacts[1])
parameters = [θb1; θb2; θc1]

x = [0,0,0, 0,0,0, 1e-5, 10.52]
solver.parameters .= parameters
e_expr(x, parameters)

solve!(solver)
solver.solution.all


parameters = [θb1; θb2; θc1]
ex_expr(x, parameters)

bodies





Xp2 = [[+0.6,1,0.0]]
Xc2 = [[-0.6,1,0.0]]
Vp15 = [[-1,0,0.0]]
Vc15 = [[+1,0,0.0]]
Pp = []
iter = []
H = 100

for i = 1:H
    bodies[1].pose .= Xp2[end]
    bodies[2].pose .= Xc2[end]

    bodies[1].velocity .= Vp15[end]
    bodies[2].velocity .= Vc15[end]

    θb1 = get_parameters(bodies[1])
    θb2 = get_parameters(bodies[2])
    θc1 = get_parameters(contacts[1])
    parameters = [θb1; θb2; θc1]

    solver.parameters .= parameters

    solve!(solver)
    solver.solution.all
    vp25 = solver.solution.all[1:3]
    vc25 = solver.solution.all[4:6]
    push!(Xp2, Xp2[end] + timestep * vp25)
    push!(Xc2, Xc2[end] + timestep * vc25)
    push!(Vp15, vp25)
    push!(Vc15, vc25)


    xl3 = zeros(131)
    contact_solver.solver.parameters .= pack_lp_parameters(Xp2[end][1:2], Xp2[end][3:3], Xc2[end][1:2], Xc2[end][3:3], Ap, bp, Ac, bc)
    solve!(contact_solver.solver)
    extract_subvariables!(xl3, contact_solver.solver)
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl3, contacts[1])
    p_parent += (Xp2[end][1:2] + Xc2[end][1:2])/2
    push!(Pp, p_parent)
    push!(iter, solver.trace.iterations)
end
scatter(iter)


################################################################################
# visualization
################################################################################
set_floor!(vis)
set_background!(vis)
set_light!(vis)

using Plots
build_2d_polytope!(vis, Ap, bp, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ac, bc, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_2d_polytope!(vis, Xp2[i][1:2], Xp2[i][3:3], name=:polya)
        set_2d_polytope!(vis, Xc2[i][1:2], Xc2[i][3:3], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Pp[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
render(vis)
