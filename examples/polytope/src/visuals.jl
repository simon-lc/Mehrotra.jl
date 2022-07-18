######################################################################
# polytope
######################################################################
function build_polytope!(vis::Visualizer, A::Matrix{T}, b::Vector{T};
        name::Symbol=:polytope,
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    h = hrep(A, b)
    p = polyhedron(h)
    m = Polyhedra.Mesh(p)
    setobject!(vis[name], m, MeshPhongMaterial(color=color))
    return nothing
end

function build_2d_polytope!(vis::Visualizer, A::Matrix{T}, b::Vector{T};
        name::Symbol=:polytope,
        color=RGBA(0.8, 0.8, 0.8, 1.0)) where T

    n = size(A)[1]
    Ae = [zeros(n) A]
    Ae = [Ae;
         -1 0 0;
          1 0 0]
    be = [b; 0.1; 00]
    build_polytope!(vis, Ae, be, name=name, color=color)
    return nothing
end

function set_polytope!(vis::Visualizer, p::Vector{T}, q::Vector{T};
        name::Symbol=:polytope) where T

    settransform!(vis[name], MeshCat.compose(
        MeshCat.Translation(p...),
        MeshCat.LinearMap(z_rotation(q)),
        )
    )
    return nothing
end

function set_2d_polytope!(vis::Visualizer, p::Vector{T}, q::Vector{T};
        name::Symbol=:polytope) where T
    pe = [0; p]

    settransform!(vis[name], MeshCat.compose(
        MeshCat.Translation(SVector{3}(pe)),
        MeshCat.LinearMap(rotationmatrix(RotX(q[1]))),
        )
    )
    return nothing
end

######################################################################
# body
######################################################################
function build_body!(vis::Visualizer, body::Body182;
        name::Symbol=body.name, 
        collider_color=RGBA(0.2, 0.2, 0.2, 0.8),
        center_of_mass_color=RGBA(1, 1, 1, 1.0),
        center_of_mass_radius=0.025,
        ) where T

    # colliders
    num_colliders = length(body.b_colliders)
    for i = 1:num_colliders
        A = body.A_colliders[i]
        b = body.b_colliders[i]
        build_2d_polytope!(vis[:bodies][name], A, b, name=Symbol(i), color=collider_color)
    end
    
    # center of mass
    setobject!(vis[:bodies][name][:com],
        HyperSphere(GeometryBasics.Point(0,0,0.), center_of_mass_radius),
        MeshPhongMaterial(color=center_of_mass_color));
    return nothing
end

function set_body!(vis::Visualizer, body::Body182, pose; name=body.name)
    p = pose[1:2]
    q = pose[3:3]
    pe = [0; p]
    settransform!(vis[:bodies][name], MeshCat.compose(
        MeshCat.Translation(SVector{3}(pe)),
        MeshCat.LinearMap(rotationmatrix(RotX(q[1]))),
        )
    )
    return nothing
end

######################################################################
# 2D frame
######################################################################
function build_2d_frame!(vis::Visualizer;
    name::Symbol=:contact, 
    origin_color=RGBA(0.2, 0.2, 0.2, 0.8),
    normal_axis_color=RGBA(0, 1, 0, 0.8),
    tangent_axis_color=RGBA(1, 0, 0, 0.8),
    origin_radius=0.025,
    ) where T

    # axes
    build_rope(vis[:contacts][name]; 
        N=1, 
        color=tangent_axis_color,
        rope_type=:cylinder, 
        rope_radius=origin_radius/2, 
        name=:tangent)

    build_rope(vis[:contacts][name]; 
        N=1, 
        color=normal_axis_color,
        rope_type=:cylinder, 
        rope_radius=origin_radius/2, 
        name=:normal)

    # origin
    setobject!(vis[:contacts][name][:origin],
        HyperSphere(GeometryBasics.Point(0,0,0.), origin_radius),
        MeshPhongMaterial(color=origin_color));
    return nothing
end

function set_2d_frame!(vis::Visualizer, contact, origin, normal, tangent; name=contact.name)
    settransform!(vis[:contacts][name][:origin], 
        MeshCat.Translation(SVector{3}(0, origin...)))
    set_straight_rope(vis[:contacts][name], [0; origin], [0; origin+normal]; N=1, name=:normal)
    set_straight_rope(vis[:contacts][name], [0; origin], [0; origin+tangent]; N=1, name=:tangent)
    return nothing
end

######################################################################
# mechanism
######################################################################
function build_mechanism!(vis::Visualizer, mechanism::Mechanism182)
    for body in mechanism.bodies
        build_body!(vis, body)
    end    
    for contact in mechanism.contacts
        build_2d_frame!(vis, name=contact.name)
    end
    return nothing
end

function set_mechanism!(vis::Visualizer, mechanism::Mechanism182, storage::Storage116, i::Int)
    for (j,body) in enumerate(mechanism.bodies)
        set_body!(vis, body, storage.x[i][j])
    end
    for (j, contact) in enumerate(mechanism.contacts)
        origin = storage.contact_point[i][j]
        normal = storage.normal[i][j]
        tangent = storage.tangent[i][j]
        set_2d_frame!(vis, contact, origin, normal, tangent)
    end
    return nothing
end

function visualize!(vis::Visualizer, mechanism::Mechanism182, storage::Storage116{T,H}; 
        animation=MeshCat.Animation(Int(floor(1/mechanism.bodies[1].timestep[1])))) where {T,H}

    build_mechanism!(vis, mechanism)
    for i = 1:H
        atframe(animation, i) do
            set_mechanism!(vis, mechanism, storage, i)
        end
    end
    MeshCat.setanimation!(vis, animation)
    return vis, animation
end

