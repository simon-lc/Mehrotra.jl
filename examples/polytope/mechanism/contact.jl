################################################################################
# contact
################################################################################
struct Contact171{T,D,NP,NC}
    name::Symbol
    node_index::NodeIndices171
    contact_solver::ContactSolver171
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function Contact171(
        Ap::Matrix{T},
        bp::Vector{T},
        Ac::Matrix{T},
        bc::Vector{T};
        name::Symbol=:contact,
        node_index::NodeIndices171=NodeIndices171()) where {T}

    contact_solver = ContactSolver(Ap, bp, Ac, bc)
    d = size(Ap, 2)
    np = size(Ap, 1)
    nc = size(Ac, 1)
    return Contact171{T,d,np,nc}(
        name,
        node_index,
        contact_solver,
        Ap,
        bp,
        Ac,
        bc,
    )
end

function Contact171(parent_body::Body171, child_body::Body171) where {T}
    return Contact171(
        parent_body.A_colliders[1],
        parent_body.b_colliders[1],
        child_body.A_colliders[1],
        child_body.b_colliders[1],
    )
end

function variable_dimension(contact::Contact171{T,D}) where {T,D}
    if D == 2
        nγ = 2*1 # impact (dual and slack)
        # nb = 2*2 # friction (dual and slack)
        nx = nγ# + nb
    else
        error("no 3D yet")
    end
    return nx
end

function equality_dimension(contact::Contact171{T,D}) where {T,D}
    if D == 2
        nγ = 1 # impact slackness equality constraint
        # nb = 2 # friction slackness equality constraints
        ne = nγ# + nb
    else
        error("no 3D yet")
    end
    return ne
end

function parameter_dimension(contact::Contact171{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = nAp + nbp + nAc + nbc
    return nθ
end

function subparameter_dimension(contact::Contact171{T,D,NP,NC}) where {T,D,NP,NC}
    nθl = contact.contact_solver.num_parameters
    return nθl
end

function subvariable_dimension(contact::Contact171{T,D,NP,NC}) where {T,D,NP,NC}
    nxl = contact.contact_solver.num_outparameters
    return nxl
end

function unpack_contact_variables(x::Vector{T}) where T
    off = 0
    γ = x[off .+ (1:1)]; off += 1
    sγ = x[off .+ (1:1)]; off += 1
    return γ, sγ
end

function get_parameters(contact::Contact171{T,D}) where {T,D}
    θ = [
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::Contact171{T,D,NP,NC}, θ) where {T,D,NP,NC}
    off = 0
    contact.A_parent_collider .= reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    contact.b_parent_collider .= θ[off .+ (1:NP)]; off += NP
    contact.A_child_collider .= reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    contact.b_child_collider .= θ[off .+ (1:NC)]; off += NC
    return nothing
end

function unpack_contact_parameters(θ::Vector, contact::Contact171{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    A_parent_collider = reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    b_parent_collider = θ[off .+ (1:NP)]; off += NP
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return A_parent_collider, b_parent_collider, A_child_collider, b_child_collider
end

function unpack_contact_subvariables(xl::Vector, contact::Contact171{T,D,NP,NC}) where {T,D,NP,NC}
    nθl = subparameter_dimension(contact)

    off = 0
    ϕ = xl[off .+ (1:1)]; off += 1
    p_parent = xl[off .+ (1:D)]; off += D
    p_child = xl[off .+ (1:D)]; off += D
    N = reshape(xl[off .+ (1:2D+2)], (1,2D+2)); off += 2D+2
    ∂p_parent = reshape(xl[off .+ (1:D*nθl)], (D,nθl)); off += D*nθl
    ∂p_child = reshape(xl[off .+ (1:D*nθl)], (D,nθl)); off += D*nθl
    return ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child
end

function contact_residual!(e, x, xl, θ, contact::Contact171, pbody::Body171, cbody::Body171)
    # variables
    γ, sγ = unpack_contact_variables(x[contact.node_index.x])
    # subvariables
    ϕ, p_parent, p_child, N, ∂p_parent, ∂p_child = unpack_contact_subvariables(xl, contact)

    # dynamics
    e[contact.node_index.e] .+= sγ - (ϕ .- 0.1)
    e[[pbody.node_index.e; cbody.node_index.e]] .+= -Nq2'*γ
    return nothing
end

function contact_residual!(e, x, θ, contact::Contact171, pbody::Body171, cbody::Body171)

    contact_solver = contact.contact_solver

    xp2, vp15, up2, timestep_p = unpack_body_parameters(θ[pbody.node_index.θ], pbody)
    xc2, vc15, uc2, timestep_c = unpack_body_parameters(θ[cbody.node_index.θ], cbody)
    vp25 = unpack_body_variables(x[pbody.node_index.x])
    vc25 = unpack_body_variables(x[cbody.node_index.x])
    xp3 = xp2 + timestep_p[1] * vp25
    xc3 = xc2 + timestep_c[1] * vc25

    # subvariables
    set_pose_parameters!(contact_solver.solver, xp2, xc2)
    update_outvariables!(contact_solver, contact_solver.solver.parameters)
    ϕ2, p2_parent, p2_child, N2, ∂p2_parent, ∂p2_child =
        unpack_contact_subvariables(contact_solver.outvariables, contact)

    # subvariables
    set_pose_parameters!(contact_solver.solver, xp3, xc3)
    update_outvariables!(contact_solver, contact_solver.solver.parameters)
    ϕ3, p3_parent, p3_child, N3, ∂p3_parent, ∂p3_child =
        unpack_contact_subvariables(contact_solver.outvariables, contact)

    # variables
    γ, sγ = unpack_contact_variables(x[contact.node_index.x])

    # dynamics
    e[contact.node_index.e] .+= sγ - (ϕ3 .- 0.0)
    e[[pbody.node_index.e; cbody.node_index.e]] .+= -N2'*γ
    return nothing
end
