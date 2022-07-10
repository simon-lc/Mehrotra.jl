################################################################################
# contact
################################################################################
struct Friction171{T,D,NP,NC}
    name::Symbol
    node_index::NodeIndices171
    contact_solver::ContactSolver171
    μ::Vector{T}
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function Friction171(
        μ::Vector{T},
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
    return Friction171{T,d,np,nc}(
        name,
        node_index,
        contact_solver,
        μ,
        Ap,
        bp,
        Ac,
        bc,
    )
end

function Friction171(parent_body::Body171, child_body::Body171, μ) where {T}
    return Friction171(
        μ,
        parent_body.A_colliders[1],
        parent_body.b_colliders[1],
        child_body.A_colliders[1],
        child_body.b_colliders[1],
    )
end

function variable_dimension(contact::Friction171{T,D}) where {T,D}
    if D == 2
        nγ = 2*1 # impact (dual and slack)
        nψ = 2*1 # friction cone (dual and slack)
        nβ = 2*2 # friction optimality (dual and slack)
        nx = nγ + nψ + nβ
    else
        error("no 3D yet")
    end
    return nx
end

function equality_dimension(contact::Friction171{T,D}) where {T,D}
    if D == 2
        nγ = 1*1 # impact (dual and slack)
        nψ = 1*1 # friction cone (dual and slack)
        nβ = 1*2 # friction optimality (dual and slack)
        ne = nγ + nψ + nβ
    else
        error("no 3D yet")
    end
    return ne
end

function parameter_dimension(contact::Friction171{T,D}) where {T,D}
    nμ = 1
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = nμ + nAp + nbp + nAc + nbc
    return nθ
end

function subparameter_dimension(contact::Friction171{T,D,NP,NC}) where {T,D,NP,NC}
    nθl = contact.contact_solver.num_parameters
    return nθl
end

function subvariable_dimension(contact::Friction171{T,D,NP,NC}) where {T,D,NP,NC}
    nxl = contact.contact_solver.num_outparameters
    return nxl
end

function unpack_contact_variables(x::Vector{T}) where T
    off = 0
    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2
    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    return γ, ψ, β, sγ, sψ, sβ
end

function get_parameters(contact::Friction171{T,D}) where {T,D}
    θ = [
        contact.μ;
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::Friction171{T,D,NP,NC}, θ) where {T,D,NP,NC}
    off = 0
    contact.μ .= θ[off .+ (1:1)]; off += 1
    contact.A_parent_collider .= reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    contact.b_parent_collider .= θ[off .+ (1:NP)]; off += NP
    contact.A_child_collider .= reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    contact.b_child_collider .= θ[off .+ (1:NC)]; off += NC
    return nothing
end

function unpack_contact_parameters(θ::Vector, contact::Friction171{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    μ = θ[off .+ (1:1)]; off += 1
    A_parent_collider = reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    b_parent_collider = θ[off .+ (1:NP)]; off += NP
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return μ, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider
end

function unpack_contact_subvariables(xl::Vector, contact::Friction171{T,D,NP,NC}) where {T,D,NP,NC}
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

function x_2d_rotation(q)
    # wRb = x_2d_rotation(q)
    # bRw = x_2d_rotation(q)'
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

function contact_residual!(e, x, θ, contact::Friction171, pbody::Body171, cbody::Body171)

    contact_solver = contact.contact_solver

    xp2, vp15, up2, timestep_p = unpack_body_parameters(θ[pbody.node_index.θ], pbody)
    xc2, vc15, uc2, timestep_c = unpack_body_parameters(θ[cbody.node_index.θ], cbody)
    vp25 = unpack_body_variables(x[pbody.node_index.x])
    vc25 = unpack_body_variables(x[cbody.node_index.x])
    xp3 = xp2 + timestep_p[1] * vp25
    xc3 = xc2 + timestep_c[1] * vc25
    xp4 = xp2 + 2timestep_p[1] * vp25
    xc4 = xc2 + 2timestep_c[1] * vc25

    # subvariables
    set_pose_parameters!(contact_solver.solver, xp3, xc3)
    update_outvariables!(contact_solver, contact_solver.solver.parameters)
    ϕ3, p3_parent, p3_child, N3, ∂p3_parent, ∂p3_child =
        unpack_contact_subvariables(contact_solver.outvariables, contact)

    # contact point in the word frame
    pw = p3_parent + (xp3[1:2] + xc3[1:2])/2
    # contact point in their respective body frames
    pp = x_2d_rotation(xp3[3:3])' * (pw - xp3[1:2])
    pc = x_2d_rotation(xc3[3:3])' * (pw - xc3[1:2])
    # contact point velocities in the world frame when attached to their respective bodies
    S = [0 1; -1 0]
    vp = vp25[1:2] + vp25[3] * S * (xp3[1:2] - pw)
    vc = vc25[1:2] + vc25[3] * S * (xc3[1:2] - pw)
    np = normalize(N3[1:2])
    nc = normalize(N3[4:5])
    # velocity projected onto the tangential plane
    vtp = (S * np)' * vp
    vtc = (S * nc)' * vc
    # velocity mapping
    Dp = (S * np)' * ([1 0 0; 0 1 0] + [zeros(2,2)  S * (xp3[1:2] - pw)])
    Dc = (S * nc)' * ([1 0 0; 0 1 0] + [zeros(2,2)  S * (xc3[1:2] - pw)])
    D = [Dp Dc]

    # variables
    # @show typeof(x)
    # @show typeof(contact)
    # @show typeof(contact.node_index)
    γ, ψ, β, sγ, sψ, sβ = unpack_contact_variables(x[contact.node_index.x])

    # jacobians
    # D = ∂ϕt / ∂q
    P = [+D;
         -D]

    # dynamics
    e[contact.node_index.e] .+= [
        sγ - (ϕ3 .- 0.0);
        sψ - (μ .* γ - [sum(β)]);
        sβ - (P * [vp25; vc25] + ψ[1] * ones(2));
        ]

    e[[pbody.node_index.e; cbody.node_index.e]] .+= -N3'*γ - P'*β
    return nothing
end
