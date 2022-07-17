using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions
using Plots


include("../polytope.jl")
include("../visuals.jl")
include("../rotate.jl")
include("../quaternion.jl")

vis = Visualizer()
render(vis)

include("node.jl")
include("body.jl")
include("poly_poly.jl")
include("poly_halfspace.jl")
include("mechanism.jl")
include("simulate.jl")



################################################################################
# demo
################################################################################
# parameters
Af = [0.0  +1.0]
bf = [0.0]
Ap2 = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.20ones(4,2);
bp2 = 0.2*[
    -0.5,
    +1,
    +1.5,
     1,
    ];  
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.30ones(4,2);
bp = 0.2*[
    +1,
    +1,
    +1,
     1,
    ];
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.20ones(4,2);
bc = 0.2*[
     1,
     1,
     1,
     1,
    ];
build_2d_polytope!(vis, Ap, bp, name=:pbody, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ap2, bp2, name=:pbody2, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ac, bc, name=:cbody, color=RGBA(0.9,0.9,0.9,0.7))

timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

# nodes
pbody = Body181(timestep, mass, inertia, [Ap, Ap2], [bp, bp2], gravity=+gravity, name=:pbody);
cbody = Body181(timestep, 1e1*mass, 1e1*inertia, [Ac], [bc], gravity=+gravity, name=:cbody);
bodies = [pbody, cbody];
contacts = [
    PolyPoly181(bodies[1], bodies[2], friction_coefficient=0.9, name=:contact),
    PolyPoly181(bodies[1], bodies[2], parent_collider_id=2, friction_coefficient=0.9, name=:contact2),
    PolyHalfSpace181(bodies[1], Af, bf, friction_coefficient=0.9, name=:phalfspace),
    PolyHalfSpace181(bodies[2], Af, bf, friction_coefficient=0.9, name=:chalfspace),
    PolyHalfSpace181(bodies[1], Af, bf, parent_collider_id=2, friction_coefficient=0.9, name=:p2halfspace),
    ]
indexing!([bodies; contacts])

local_mechanism_residual(primals, duals, slacks, parameters) = 
    mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

options=Options(
    # verbose=false,#true, 
    verbose=true, 
    complementarity_tolerance=1e-4,
    compressed_search_direction=false, 
    max_iterations=30,
    sparse_solver=false,
    warm_start=false,
)
mech = Mechanism181(local_mechanism_residual, bodies, contacts, options=options)

initialize_solver!(mech.solver)
solve!(mech.solver)


################################################################################
# test simulation
################################################################################
Xp2 = [[+0.1,3.0,+1.0]]
Xc2 = [[-0.1,1.0,-1.0]]
Vp15 = [[-0,0,-0.0]]
Vc15 = [[+0,0,+0.0]]
C = [[[0,0.0]] for i=1:3]
Np = [[[0,1.0]] for i=1:3]
Nc = [[[0,1.0]] for i=1:3]
Tp = [[[1,0.0]] for i=1:3]
Tc = [[[0,1.0]] for i=1:3]
iter = []
solutions = []

xp2 = [+0.1,3.0,+1.0]
xc2 = [-0.1,1.0,-1.0]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]
u0 = zeros(6)
H0 = 10
storage = simulate!(mech, z0, H0);

function step!(mechanism::Mechanism181, z0, u)
    set_current_state!(mechanism, z0)
    set_input!(mechanism, u)
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

function step!(mechanism::Mechanism181, z0; controller::Function=m->nothing)
    set_current_state!(mechanism, z0)
    controller(mechanism) # sets the control inputs u
    update_parameters!(mechanism)
    solve!(mechanism.solver)
    z1 = get_next_state(mechanism)
    return z1
end

mutable struct Storage112{T,H}
    z::Vector{Vector{T}} # H x nz
    u::Vector{Vector{T}} # H x nu
    x::Vector{Vector{Vector{T}}} # H x nb x nx
    v::Vector{Vector{Vector{T}}} # H x nb x nv
end

function Storage(dim::MechanismDimensions181, H::Int, T=Float64)
    z = [zeros(T, dim.state) for i = 1:H]
    u = [zeros(T, dim.input) for i = 1:H]
    x = [[zeros(T, dim.body_configuration) for j = 1:dim.bodies] for i = 1:H]
    v = [[zeros(T, dim.body_velocity) for j = 1:dim.bodies] for i = 1:H]
    storage = Storage112{T,H}(z, u, x, v)
    return storage
end

function simulate!(mechanism::Mechanism181{T}, z0, H::Int; 
        controller::Function=(m,i)->nothing) where T

    storage = Storage(mechanism.dimensions, H, T)
    z = copy(z0)
    for i = 1:H
        z .= step!(mechanism, z, controller=m -> controller(m,i))
        record!(storage, mechanism, i)
    end
    return storage
end

function record!(storage::Storage112{T,H}, mechanism::Mechanism181{T,D,NB}, i::Int) where {T,H,D,NB}
    storage.z[i] .= get_current_state(mechanism)
    storage.u[i] .= get_input(mechanism)
    for j = 1:NB
        storage.x[i][j] .= mechanism.bodies[j].pose
        storage.v[i][j] .= mechanism.bodies[j].velocity
    end
    return nothing
end




H = 55
Up = [zeros(3) for i=1:H]
Uc = [zeros(3) for i=1:H]
for i = 1:H
    bodies[1].pose .= Xp2[end]
    bodies[1].velocity .= Vp15[end]
    bodies[1].input .= Up[i]

    bodies[2].pose .= Xc2[end]
    bodies[2].velocity .= Vc15[end]
    bodies[2].input .= Uc[i]

    parameters = vcat(get_parameters.(nodes)...)
    solver.parameters .= parameters
    
    solve!(solver)
    x = deepcopy(solver.solution.all)
    push!(solutions, deepcopy(x))
    push!(iter, solver.trace.iterations)

    # bodies
    vp25 = unpack_variables(x[bodies[1].index.variables], bodies[1])
    vc25 = unpack_variables(x[bodies[2].index.variables], bodies[2])
    
    # push!(Vp15, dvp25 / timestep)
    # push!(Vc15, dvc25 / timestep)
    push!(Vp15, vp25)
    push!(Vc15, vc25)
    push!(Xp2, Xp2[end] + timestep * vp25)
    # push!(Xp2, Xp2[end] + dvp25)
    push!(Xc2, Xc2[end] + timestep * vc25)
    # push!(Xc2, Xc2[end] + dvc25)
    
    # contacts and halfspaces
    for (j,contact) in enumerate(contacts[1:1])
        c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contact.index.variables], contact)
        normal_pw = -x_2d_rotation(Xp2[end][3:3]) * Ap' * λp
        normal_cw = +x_2d_rotation(Xc2[end][3:3]) * Ac' * λc
        R = [0 1; -1 0]
        tangent_pw = R * normal_pw
        tangent_cw = R * normal_cw

        push!(C[j], c + (Xp2[end][1:2] + Xc2[end][1:2]) ./ 2)
        push!(Np[j], normal_pw)
        push!(Nc[j], normal_cw)
        push!(Tp[j], tangent_pw)
        push!(Tc[j], tangent_cw)
    end

    for (j,contact) in enumerate(hspaces[1:2])
        c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contact.index.variables], contact)
        normal_pw = (j==1) ? -x_2d_rotation(Xp2[end][3:3]) * Ap' * λp :
            -x_2d_rotation(Xc2[end][3:3]) * Ac' * λp

        normal_cw = +x_2d_rotation(zeros(3)[3:3]) * Af' * λc
        R = [0 1; -1 0]
        tangent_pw = R * normal_pw
        tangent_cw = R * normal_cw

        j == 1 ? push!(C[length(contacts[1:1]) + j], c + Xp2[end][1:2]) :
            push!(C[length(contacts[1:1]) + j], c + Xc2[end][1:2])

        push!(Np[length(contacts[1:1]) + j], normal_pw)
        push!(Nc[length(contacts[1:1]) + j], normal_cw)
        push!(Tp[length(contacts[1:1]) + j], tangent_pw)
        push!(Tc[length(contacts[1:1]) + j], tangent_cw)
    end

end


scatter(iter)
plot!(hcat([abs.(s[solver.indices.primals]) for s in solutions]...)', legend=false)
scatter(iter)
plot!(hcat([abs.(s[solver.indices.duals]) for s in solutions]...)', legend=false)
scatter(iter)
plot!(hcat([abs.(s[solver.indices.slacks]) for s in solutions]...)', legend=false)

solver.indices.variables
[s[1] for s in solutions]
bodies[1].index.primals

# velocity of body 1 along y
plot(hcat([abs.(s[solver.indices.primals])[2:2] for s in solutions]...)', legend=false)

plot(hcat([abs.(s[solver.indices.duals])[1:12] for s in solutions]...)', legend=false)
plot(hcat([abs.(s[solver.indices.duals])[13:21] for s in solutions]...)', legend=false)
plot(hcat([abs.(s[solver.indices.duals])[22:30] for s in solutions]...)', legend=false)
plot(hcat([abs.(s[solver.indices.duals])[22:30][3:3] for s in solutions]...)', legend=false)

solver.indices.duals
hspaces[2].index.duals

################################################################################
# visualization
################################################################################
render(vis)
set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polytope!(vis, Ap, bp, name=:pbody, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ap2, bp2, name=:pbody2, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polytope!(vis, Ac, bc, name=:cbody, color=RGBA(0.9,0.9,0.9,0.7))
for j = 1:3
    setobject!(vis[Symbol(:contact,j)],
        HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
        MeshPhongMaterial(color=RGBA(1,0,0,1.0)));

    build_rope(vis; N=1, color=Colors.RGBA(0,0,0,1),
        rope_type=:cylinder, rope_radius=0.04, name=Symbol(:normal_p,j))

    build_rope(vis; N=1, color=Colors.RGBA(1,0,0,1),
        rope_type=:cylinder, rope_radius=0.04, name=Symbol(:tangent_p,j))

    # build_rope(vis; N=1, color=Colors.RGBA(0,0,0,1),
    #     rope_type=:cylinder, rope_radius=0.04, name=Symbol(:normal_c,j))

    # build_rope(vis; N=1, color=Colors.RGBA(0,1,0,1),
    #     rope_type=:cylinder, rope_radius=0.04, name=Symbol(:tangent_c,j))
end


anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 2:H+1
    atframe(anim, i) do
        for j = 1:3
            settransform!(vis[Symbol(:contact,j)], MeshCat.Translation(SVector{3}(0, C[j][i]...)));
            set_straight_rope(vis, [0; C[j][i]], [0; C[j][i]+Np[j][i]]; N=1, name=Symbol(:normal_p,j));
            # set_straight_rope(vis, [0; C[j][i]], [0; C[j][i]+Nc[j][i]]; N=1, name=Symbol(:normal_c,j));
            set_straight_rope(vis, [0; C[j][i]], [0; C[j][i]+Tp[j][i]]; N=1, name=Symbol(:tangent_p,j));
            # set_straight_rope(vis, [0; C[j][i]], [0; C[j][i]+Tc[j][i]]; N=1, name=Symbol(:tangent_p,j));
        end
        set_2d_polytope!(vis, Xp2[i][1:2], Xp2[i][3:3], name=:pbody);
        set_2d_polytope!(vis, Xp2[i][1:2], Xp2[i][3:3], name=:pbody2);
        set_2d_polytope!(vis, Xc2[i][1:2], Xc2[i][3:3], name=:cbody);
    end;
end;
MeshCat.setanimation!(vis, anim)
# open(vis)
# convert_frames_to_video_and_gif("polytope_drop_slow")

# ex = solver.data.jacobian_variables_dense
# plot(Gray.(abs.(ex)))
# plot(Gray.(abs.(ex - ex')))
# plot(Gray.(abs.(ex + ex')))
# plot(Gray.(1e3abs.(solver.data.jacobian_variables_dense)))

# scatter(solver.solution.all)
# scatter(solver.solution.primals)
# scatter(solver.solution.duals)
# scatter(solver.solution.slacks)

# plot(hcat(solutions...)', legend=false)
