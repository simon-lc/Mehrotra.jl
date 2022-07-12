################################################################################
# contact
################################################################################
struct Contact174{T,D,NP,NC}
    name::Symbol
    parent_name::Symbol
    child_name::Symbol
    node_index::NodeIndices174
    contact_solver::ContactSolver174
    friction_coefficient::Vector{T}
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function Contact174(parent_body::Body174{T}, child_body::Body174{T};
        name::Symbol=:contact, friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    child_name = child_body.name
    Ap = parent_body.A_colliders[1]
    bp = parent_body.b_colliders[1]
    Ac = child_body.A_colliders[1]
    bc = child_body.b_colliders[1]

    return Contact174(parent_name, child_name, friction_coefficient, Ap, bp, Ac, bc;
        name=name)
end

function Contact174(
        parent_name::Symbol,
        child_name::Symbol,
        friction_coefficient,
        Ap::Matrix{T},
        bp::Vector{T},
        Ac::Matrix{T},
        bc::Vector{T};
        name::Symbol=:contact) where {T}

    contact_solver = ContactSolver(Ap, bp, Ac, bc)
    d = size(Ap, 2)
    np = size(Ap, 1)
    nc = size(Ac, 1)
    node_index = NodeIndices174()
    return Contact174{T,d,np,nc}(
        name,
        parent_name,
        child_name,
        node_index,
        contact_solver,
        [friction_coefficient],
        Ap,
        bp,
        Ac,
        bc,
    )
end

primal_dimension(contact::Contact174{T,D}) where {T,D} = 0
cone_dimension(contact::Contact174{T,D}) where {T,D} = 1+1+2 # γ ψ β
variable_dimension(contact::Contact174{T,D}) where {T,D} = primal_dimension(contact) + 2 * cone_dimension(contact)
equality_dimension(contact::Contact174{T,D}) where {T,D} = primal_dimension(contact) + cone_dimension(contact)

function parameter_dimension(contact::Contact174{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = 1 + nAp + nbp + nAc + nbc
    return nθ
end

function subparameter_dimension(contact::Contact174{T,D,NP,NC}) where {T,D,NP,NC}
    nθl = contact.contact_solver.num_parameters
    return nθl
end

function subvariable_dimension(contact::Contact174{T,D,NP,NC}) where {T,D,NP,NC}
    nxl = contact.contact_solver.num_outvariables
    return nxl
end

function unpack_variables(x::Vector{T}, contact::Contact174) where T
    num_cone = cone_dimension(contact)
    off = 0
    z = x[off .+ (1:num_cone)]; off += num_cone
    s = x[off .+ (1:num_cone)]; off += num_cone
    return z, s
end

function get_parameters(contact::Contact174{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::Contact174{T,D,NP,NC}, θ) where {T,D,NP,NC}
    off = 0
    contact.friction_coefficient .= θ[off .+ (1:1)]; off += 1
    contact.A_parent_collider .= reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    contact.b_parent_collider .= θ[off .+ (1:NP)]; off += NP
    contact.A_child_collider .= reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    contact.b_child_collider .= θ[off .+ (1:NC)]; off += NC
    return nothing
end

function unpack_parameters(θ::Vector, contact::Contact174{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    A_parent_collider = reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    b_parent_collider = θ[off .+ (1:NP)]; off += NP
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return friction_coefficient, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider
end

function unpack_subvariables(xl::Vector, contact::Contact174{T,D,NP,NC}) where {T,D,NP,NC}
    nθl = subparameter_dimension(contact)

    off = 0
    ϕ = xl[off .+ (1:1)]; off += 1
    p_parent = xl[off .+ (1:D)]; off += D
    p_child = xl[off .+ (1:D)]; off += D
    N = reshape(xl[off .+ (1:2D+2)], (1,2D+2)); off += 2D+2
    ∂p_parent = reshape(xl[off .+ (1:D*nθl)], (D,nθl)); off += D*nθl
    ∂p_child = reshape(xl[off .+ (1:D*nθl)], (D,nθl)); off += D*nθl
    nw = xl[off .+ (1:D)]; off += D
    tw = xl[off .+ (1:D)]; off += D
    return ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child, nw, tw
end

# function contact_residual!(e, x, xl, θ, contact::Contact174, pbody::Body174, cbody::Body174)
#     # variables
#     z, s = unpack_variables(x[contact.node_index.x], contact)
#     # subvariables
#     ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_subvariables(xl, contact)
#
#     # dynamics
#     e[contact.node_index.e] .+= sγ - (ϕ .- 0.1)
#     e[[pbody.node_index.e; cbody.node_index.e]] .+= -Nq2'*γ
#     return nothing
# end

function contact_residual!(e, x, θ, contact::Contact174, pbody::Body174, cbody::Body174)

    contact_solver = contact.contact_solver

    # parameters
    friction_coefficient, _ = unpack_parameters(θ[contact.node_index.θ], contact)
    xp2, vp15, up2, timestep_p, _ = unpack_parameters(θ[pbody.node_index.θ], pbody)
    xc2, vc15, uc2, timestep_c, _ = unpack_parameters(θ[cbody.node_index.θ], cbody)
    vp25 = unpack_variables(x[pbody.node_index.x], pbody)
    vc25 = unpack_variables(x[cbody.node_index.x], cbody)
    xp3 = xp2 + timestep_p[1] * vp25
    xc3 = xc2 + timestep_c[1] * vc25
    xp4 = xp2 + 2timestep_p[1] * vp25
    xc4 = xc2 + 2timestep_c[1] * vc25

    # # subvariables
    # set_pose_parameters!(contact_solver.solver, xp2, xc2)
    # update_outvariables!(contact_solver, contact_solver.solver.parameters)
    # ϕ2, p2_parent, p2_child, N2, ∂p2_parent, ∂p2_child =
    #     unpack_subvariables(contact_solver.outvariables, contact)

    # subvariables
    set_pose_parameters!(contact_solver.solver, xp3, xc3)
    update_outvariables!(contact_solver, contact_solver.solver.parameters)
    ϕ3, p3_parent, p3_child, N3, ∂p3_parent, ∂p3_child, nw3, tw3 =
        unpack_subvariables(contact_solver.outvariables, contact)

    p3_parent += (xp3 + xc3)[1:2] / 2
    p3_child += (xp3 + xc3)[1:2] / 2

    # variables
    z, s = unpack_variables(x[contact.node_index.x], contact)

    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:4]
    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:4]

    # force at the contact point in the contact frame
    νc = [γ; [sum(β)]]
    # rotation matrix from contact frame to world frame
    wRc = [nw3 tw3] # n points towards the parent body, [n,t,z] forms an oriented vector basis
    # force at the contact point in the world frame
    νw = wRc * νc
    νpw = +νw # parent
    νcw = -νw # child
    # wrenches at the centers of masses
    τpw = (skew([xp3[1:2] - p3_parent; 0]) * [νpw; 0])[3:3]
    τcw = (skew([xc3[1:2] - p3_child;  0]) * [νcw; 0])[3:3]
    w = [νpw; τpw; νcw; τcw]
    # mapping the force into the generalized coordinates (at the centers of masses and in the world frame)

    vptan = vp25[1:2] + (skew([xp3[1:2]-p3_parent; 0]) * [zeros(2); vp25[3]])[1:2]
    vptan = vptan'*tw3
    vctan = vc25[1:2] + (skew([xc3[1:2]-p3_child;  0]) * [zeros(2); vc25[3]])[1:2]
    vctan = vctan'*tw3
    vtan = vptan - vctan

    # dynamics
    e[contact.node_index.e] .+= [
        sγ - ϕ3;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([vtan; +vtan] + ψ[1]*ones(2));
        ]
    e[[pbody.node_index.e; cbody.node_index.e]] .+= w # -V3' * ν
    return nothing
end
