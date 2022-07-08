

function body_residual!(e, x, θ, body::Body170)
    node_index = body.node_index
    # variables = primals = velocity
    v25 = unpack_body_variables(x[node_index.x])
    # parameters
    p2, v15, u, timestep, gravity, mass, inertia = unpack_body_parameters(θ[node_index.θ], body)
    # integrator
    p1 = p2 - timestep[1] * v15
    p3 = p2 + timestep[1] * v25
    # mass matrix
    M = Diagonal([mass[1]; mass[1]; inertia[1]])
    # dynamics
    dynamics = M * (p3 - 2*p2 + p1)/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - u * timestep[1];
    e[node_index.e] .+= dynamics
    return nothing
end


function contact_residual!(e, x, xl2, xl3, θ, contact::Contact170, pbody::Body170, cbody::Body170)
    # variables
    γ, sγ = unpack_contact_variables(x[contact.node_index.x])
    # subvariables
    _, _, _, N2, _, _ = unpack_contact_subvariables(xl2, contact)
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl3, contact)

    # dynamics
    e[contact.node_index.e] .+= sγ - (ϕ .- 0.0)
    e[[pbody.node_index.e; cbody.node_index.e]] .+= -N2'*γ
    return nothing
end

function equality_residual(x, θ, bodies, contacts, contact_solver)
    e = zeros(num_equality)
    for body in bodies
        body_residual!(e, x, θ, body)
    end
    for contact in contacts
        num_subvariables = 131
        xp3 = unpack_body_variables(x[bodies[1].node_index.x])
        xc3 = unpack_body_variables(x[bodies[2].node_index.x])
        xp2 = unpack_body_parameters(θ[bodies[1].node_index.θ], bodies[1])[1]
        xc2 = unpack_body_parameters(θ[bodies[2].node_index.θ], bodies[2])[1]

        xl2 = zeros(num_subvariables)
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
bodya = Body170(timestep, mass, inertia, [Ap], [bp], gravity=gravity, name=:bodya)
bodyb = Body170(timestep, mass, inertia, [Ac], [bc], gravity=gravity, name=:bodyb)
bodies = [bodya, bodyb]
contacts = [Contact170(bodies[1], bodies[2])]
indexing!([bodies; contacts])
contact_solver = ContactSolver(Ap, bp, Ac, bc,
    options=Options(
    verbose=false,
    complementarity_tolerance=1e-2,
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
        cone_methods::ConeMethods{B,BX,P,PX,PXI,TA},
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
        problem.equality_constraint .= methods.e(x, θ)
    end

    if equality_jacobian_variables
        # @show size(methods.ex(x, θ))
        # @show size(problem.equality_jacobian_variables)
        problem.equality_jacobian_variables .= methods.ex(x, θ)
    end

    if equality_jacobian_parameters
        problem.equality_jacobian_jacobian .= methods.eθ(x, θ)
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

struct FiniteDiffMethods113{T,E,EX,Eθ} <: AbstractProblemMethods{T}
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


bodies[1].pose .= [+10,0,0.0]
bodies[2].pose .= [-10,0,0.0]

bodies[1].velocity .= [+0,0,0.0]
bodies[2].velocity .= [-0,0,0.0]

θb1 = get_parameters(bodies[1])
θb2 = get_parameters(bodies[2])
θc1 = get_parameters(contacts[1])
parameters = [θb1; θb2; θc1]

x = [0,0,0, 0,0,0, 10, 1e-5]
solver.parameters .= parameters
e_expr(x, parameters)

solve!(solver)


ex_expr(x, parameters)
