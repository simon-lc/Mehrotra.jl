################################################################################
# contact
################################################################################
struct Halfspace175{T,D,NP}
    name::Symbol
    parent_name::Symbol
    node_index::NodeIndices175
    friction_coefficient::Vector{T}
    A_parent_collider::Matrix{T} #polytope
    b_parent_collider::Vector{T} #polytope
    A_child_collider::Matrix{T} #polytope
    b_child_collider::Vector{T} #polytope
end

function Halfspace175(parent_body::Body175{T}, Ac::AbstractMatrix, bc::AbstractVector;
        parent_collider_id::Int=1, 
        name::Symbol=:halfspace, 
        friction_coefficient=0.2) where {T}

    parent_name = parent_body.name
    Ap = parent_body.A_colliders[parent_collider_id]
    bp = parent_body.b_colliders[parent_collider_id]

    return Halfspace175(parent_name, friction_coefficient, Ap, bp, Ac, bc;
        name=name)
end

function Halfspace175(
        parent_name::Symbol,
        friction_coefficient,
        Ap::Matrix{T},
        bp::Vector{T},
        Ac::Matrix{T},
        bc::Vector{T};
        name::Symbol=:contact) where {T}

    d = size(Ap, 2)
    np = size(Ap, 1)
    nc = size(Ac, 1)
    node_index = NodeIndices175()
    return Halfspace175{T,d,np,nc}(
        name,
        parent_name,
        child_name,
        node_index,
        [friction_coefficient],
        Ap,
        bp,
        Ac,
        bc,
    )
end

primal_dimension(contact::Halfspace175{T,D}) where {T,D} = D + 1 # x, ϕ
cone_dimension(contact::Halfspace175{T,D,NP,NC}) where {T,D,NP,NC} = 1 + 1 + 2 + NP + 1 # γ ψ β λp, λc
variable_dimension(contact::Halfspace175{T,D}) where {T,D} = primal_dimension(contact) + 2 * cone_dimension(contact)
equality_dimension(contact::Halfspace175{T,D}) where {T,D} = primal_dimension(contact) + cone_dimension(contact)

function parameter_dimension(contact::Halfspace175{T,D}) where {T,D}
    nAp = length(contact.A_parent_collider)
    nbp = length(contact.b_parent_collider)
    nAc = length(contact.A_child_collider)
    nbc = length(contact.b_child_collider)
    nθ = 1 + nAp + nbp + nAc + nbc
    return nθ
end

function unpack_variables(x::Vector{T}, contact::Halfspace175{T,D,NP,NC}) where {T,D,NP,NC}
    num_cone = cone_dimension(contact)
    off = 0
    c = x[off .+ (1:2)]; off += 2
    ϕ = x[off .+ (1:1)]; off += 1

    γ = x[off .+ (1:1)]; off += 1
    ψ = x[off .+ (1:1)]; off += 1
    β = x[off .+ (1:2)]; off += 2
    λp = x[off .+ (1:NP)]; off += NP
    λc = x[off .+ (1:NC)]; off += NC

    sγ = x[off .+ (1:1)]; off += 1
    sψ = x[off .+ (1:1)]; off += 1
    sβ = x[off .+ (1:2)]; off += 2
    sp = x[off .+ (1:NP)]; off += NP
    sc = x[off .+ (1:NC)]; off += NC
    return c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc
end

function get_parameters(contact::Halfspace175{T,D}) where {T,D}
    θ = [
        contact.friction_coefficient;
        vec(contact.A_parent_collider); contact.b_parent_collider;
        vec(contact.A_child_collider); contact.b_child_collider;
        ]
    return θ
end

function set_parameters!(contact::Halfspace175{T,D,NP,NC}, θ) where {T,D,NP,NC}
    friction_coefficient, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider = 
        unpack_parameters(θ, contact)
    contact.friction_coefficient .= friction_coefficient
    contact.A_parent_collider .= A_parent_collider
    contact.b_parent_collider .= b_parent_collider
    contact.A_child_collider .= A_child_collider
    contact.b_child_collider .= b_child_collider
    return nothing
end

function unpack_parameters(θ::Vector, contact::Halfspace175{T,D,NP,NC}) where {T,D,NP,NC}
    @assert D == 2
    off = 0
    friction_coefficient = θ[off .+ (1:1)]; off += 1
    A_parent_collider = reshape(θ[off .+ (1:NP*D)], (NP,D)); off += NP*D
    b_parent_collider = θ[off .+ (1:NP)]; off += NP
    A_child_collider = reshape(θ[off .+ (1:NC*D)], (NC,D)); off += NC*D
    b_child_collider = θ[off .+ (1:NC)]; off += NC
    return friction_coefficient, A_parent_collider, b_parent_collider, A_child_collider, b_child_collider
end

function contact_residual!(e, x, θ, contact::Halfspace175{T,D,NP,NC}, 
        pbody::Body175, cbody::Body175) where {T,D,NP,NC}
    
    # unpack parameters
    friction_coefficient, Ap, bp, Ac, bc = unpack_parameters(θ[contact.node_index.θ], contact)
    pp2, vp15, up2, timestep_p, gravity_p, mass_p, inertia_p = unpack_parameters(θ[pbody.node_index.θ], pbody)
    pc2, vc15, uc2, timestep_c, gravity_c, mass_c, inertia_c = unpack_parameters(θ[cbody.node_index.θ], cbody)
    
    # unpack variables
    c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contact.node_index.x], contact)
    vp25 = unpack_variables(x[pbody.node_index.x], pbody)
    vc25 = unpack_variables(x[cbody.node_index.x], cbody)
    pp3 = pp2 + timestep_p[1] * vp25
    pc3 = pc2 + timestep_c[1] * vc25
    pp1 = pp2 - timestep_p[1] * vp15
    pc1 = pc2 - timestep_c[1] * vc15

    # contact position in the world frame
    contact_w = c + (pp3 + pc3)[1:2] / 2
    # contact_p is expressed in pbody's frame
    contact_p = x_2d_rotation(pp3[3:3])' * (contact_w - pp3[1:2])
    # contact_c is expressed in cbody's frame
    contact_c = x_2d_rotation(pc3[3:3])' * (contact_w - pc3[1:2])

    # contact normal and tangent in the world frame
    normal_pw = -x_2d_rotation(pp3[3:3]) * Ap' * λp
    normal_cw = +x_2d_rotation(pc3[3:3]) * Ac' * λc
    R = [0 1; -1 0]
    tangent_pw = R * normal_pw
    tangent_cw = R * normal_cw

    # rotation matrix from contact frame to world frame
    wRp = [tangent_pw normal_pw] # n points towards the parent body, [t,n,z] forms an oriented vector basis
    wRc = [tangent_cw normal_cw] # n points towards the parent body, [t,n,z] forms an oriented vector basis

    # force at the contact point in the contact frame
    f = [β[1] - β[2]; γ]
    # force at the contact point in the world frame
    f_pw = +wRp * f # parent
    f_cw = -wRc * f # child
    # torques at the centers of masses in world frame
    τ_pw = (skew([contact_w - pp3[1:2]; 0]) * [f_pw; 0])[3:3]
    τ_cw = (skew([contact_w - pc3[1:2]; 0]) * [f_cw; 0])[3:3]
    # overall wrench on both bodies in world frame
    # mapping the contact force into the generalized coordinates (at the centers of masses and in the world frame)
    wrench_p = [f_pw; τ_pw]
    wrench_c = [f_cw; τ_cw]

    # tangential velocities at the contact point
    tanvel_p = vp25[1:2] + (skew([pp3[1:2] - contact_w; 0]) * [zeros(2); vp25[3]])[1:2]
    tanvel_p = tanvel_p' * tangent_pw
    tanvel_c = vc25[1:2] + (skew([pc3[1:2] - contact_w; 0]) * [zeros(2); vc25[3]])[1:2]
    tanvel_c = tanvel_c' * tangent_cw
    tanvel = tanvel_p - tanvel_c

    # contact equality
    residual = [
        x_2d_rotation(pp3[3:3]) * Ap' * λp + x_2d_rotation(pc3[3:3]) * Ac' * λc;
        1 - sum(λp) - sum(λc);
        sγ - ϕ;
        sψ - (friction_coefficient[1] * γ - [sum(β)]);
        sβ - ([+tanvel; -tanvel] + ψ[1]*ones(2));
        sp - (- Ap * contact_p + bp + ϕ .* ones(NP));
        sc - (- Ac * contact_c + bc + ϕ .* ones(NC));
    ]

    # fill the equality vector (residual of the equality constraints)
    e[contact.node_index.e] .+= residual
    e[pbody.node_index.e] .-= wrench_p
    e[cbody.node_index.e] .-= wrench_c
    return nothing
end

