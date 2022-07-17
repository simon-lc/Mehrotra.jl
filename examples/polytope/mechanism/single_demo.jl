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
pbody = Body182(timestep, mass, inertia, [Ap, Ap2], [bp, bp2], gravity=+gravity, name=:pbody);
cbody = Body182(timestep, 1e1*mass, 1e1*inertia, [Ac], [bc], gravity=+gravity, name=:cbody);
bodies = [pbody, cbody];
contacts = [
    PolyPoly182(bodies[1], bodies[2], friction_coefficient=0.9, name=:contact),
    PolyPoly182(bodies[1], bodies[2], parent_collider_id=2, friction_coefficient=0.9, name=:contact2),
    PolyHalfSpace182(bodies[1], Af, bf, friction_coefficient=0.9, name=:phalfspace),
    PolyHalfSpace182(bodies[2], Af, bf, friction_coefficient=0.9, name=:chalfspace),
    PolyHalfSpace182(bodies[1], Af, bf, parent_collider_id=2, friction_coefficient=0.9, name=:p2halfspace),
    ]
indexing!([bodies; contacts])

local_mechanism_residual(primals, duals, slacks, parameters) = 
    mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

options=Options(
    # verbose=false,#true, 
    # verbose=true, 
    complementarity_tolerance=1e-4,
    compressed_search_direction=false, 
    max_iterations=30,
    sparse_solver=false,
    warm_start=false,
)
mech = Mechanism182(local_mechanism_residual, bodies, contacts, options=options)

initialize_solver!(mech.solver)
solve!(mech.solver)


################################################################################
# test simulation
################################################################################
xp2 = [+0.1,3.0,+1.0]
xc2 = [-0.1,1.0,-1.0]
vp15 = [-0,0,-0.0]
vc15 = [+0,0,+0.0]
z0 = [xp2; vp15; xc2; vc15]
u0 = zeros(6)
H0 = 20
storage = simulate!(mech, z0, H0);

storage.contact_point

scatter(storage.iterations)
plot(hcat(storage.variables...)')






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
