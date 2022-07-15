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
include("contact.jl")
include("halfspace_contact.jl")
include("mechanism.jl")



################################################################################
# demo
################################################################################
# parameters
Af = [0.0  +1.0]
bf = [0.0]
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2);
bp = 0.5*[
    +1,
    +1,
    +1,
     1,
     # 2,
    ];
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.00ones(4,2);
bc = 0.4*[
     1,
     1,
     1,
     1,
    ];

timestep = 0.05;
gravity = -9.81;
mass = 1.0;
inertia = 0.2 * ones(1,1);

# nodes
pbody = Body177(timestep, mass, inertia, [Ap], [bp], gravity=+gravity, name=:pbody);
cbody = Body177(timestep, 1e1*mass, 1e1*inertia, [Ac], [bc], gravity=+gravity, name=:cbody);
bodies = [pbody, cbody];
contacts = [Contact177(bodies[1], bodies[2], friction_coefficient=0.3, name=:contact)]
# contacts = []
hspaces = [
    Halfspace177(bodies[1], Af, bf, friction_coefficient=0.3, name=:phalfspace),
    Halfspace177(bodies[2], Af, bf, friction_coefficient=0.3, name=:phalfspace)]
# hspaces = []
indexing!([bodies; contacts; hspaces])

bodies[1].index
bodies[2].index
contacts[1].index


function mechanism_residual(primals, duals, slacks, parameters, bodies, contacts, halfspaces)
    num_duals = length(duals)
    num_primals = length(primals)
    num_equality = num_primals + num_duals

    e = zeros(num_equality)
    x = [primals; duals; slacks]
    θ = parameters

    # body
    for body in bodies
        body_residual!(e, x, θ, body)
    end

    # contact
    for contact in contacts
        pbody = find_body(bodies, contact.parent_name)
        cbody = find_body(bodies, contact.child_name)
        contact_residual!(e, x, θ, contact, pbody, cbody)
    end

    # halfspace
    for contact in halfspaces
        pbody = find_body(bodies, contact.parent_name)
        contact_residual!(e, x, θ, contact, pbody)
    end
    return e
end

nodes = [bodies; contacts; hspaces];
num_primals = sum(primal_dimension.(nodes))
num_cone = sum(cone_dimension.(nodes))
primals = ones(num_primals);
duals = ones(num_cone);
slacks = ones(num_cone);
parameters = vcat(get_parameters.(nodes)...);
e = mechanism_residual(primals, duals, slacks, parameters, bodies, contacts, hspaces);

local_mechanism_residual(primals, duals, slacks, parameters) = 
    mechanism_residual(primals, duals, slacks, parameters, bodies, contacts, hspaces)

solver = Solver(
    local_mechanism_residual,
    num_primals,
    num_cone,
    parameters=parameters,
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    method_type=:finite_difference,
    options=Options(
        # verbose=false,#true, 
        verbose=true, 
        complementarity_tolerance=1e-5,
        compressed_search_direction=false, 
        max_iterations=30,
        sparse_solver=false,
        warm_start=false,
    ));


solver.indices

solve!(solver)
# solver.problem.equality_constraint[1:20]
# solver.problem.cone_product[1:20]

# initialize_primals!(solver)
# initialize_duals!(solver)
# initialize_slacks!(solver)
# initialize_interior_point!(solver)


# evaluate!(solver.problem,
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
#         sparse_solver=true,
#         ) 

# norm(solver.problem.equality_constraint)

# solver.problem.equality_constraint
# solver.problem.equality_jacobian_variables_sparse
# solver.problem.equality_jacobian_parameters_sparse
# solver.problem.cone_product
# solver.problem.cone_product_jacobian_duals_sparse
# solver.problem.cone_product_jacobian_slacks_sparse
# solver.problem.cone_product_jacobian_inverse_duals_sparse
# solver.problem.cone_product_jacobian_inverse_slacks_sparse
# solver.problem.cone_target


# norm(solver.problem.equality_constraint, Inf)
# norm(solver.problem.equality_jacobian_variables_sparse, Inf)
# norm(solver.problem.equality_jacobian_parameters_sparse, Inf)
# norm(solver.problem.cone_product, Inf)
# norm(solver.problem.cone_product_jacobian_duals_sparse, Inf)
# norm(solver.problem.cone_product_jacobian_slacks_sparse, Inf)
# norm(solver.problem.cone_product_jacobian_inverse_duals_sparse, Inf)
# norm(solver.problem.cone_product_jacobian_inverse_slacks_sparse, Inf)
# norm(solver.problem.cone_target, Inf)



################################################################################
# test simulation
################################################################################
Xp2 = [[+0.1,5.0,+1.0]]
Xc2 = [[-0.1,2.0,-1.0]]
Vp15 = [[-0,0,-0.0]]
Vc15 = [[+0,0,+0.0]]
C = [[[0,0.0]] for i=1:3]
Np = [[[0,1.0]] for i=1:3]
Nc = [[[0,1.0]] for i=1:3]
Tp = [[[1,0.0]] for i=1:3]
Tc = [[[0,1.0]] for i=1:3]
iter = []
solutions = []


H = 35
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
    
    push!(Vp15, vp25)
    push!(Vc15, vc25)
    push!(Xp2, Xp2[end] + timestep * vp25)
    push!(Xc2, Xc2[end] + timestep * vc25)
    
    # contacts and halfspaces
    for (j,contact) in enumerate(contacts)
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

    for (j,contact) in enumerate(hspaces)
        c, ϕ, γ, ψ, β, λp, λc, sγ, sψ, sβ, sp, sc = unpack_variables(x[contact.index.variables], contact)
        normal_pw = (j==1) ? -x_2d_rotation(Xp2[end][3:3]) * Ap' * λp :
            -x_2d_rotation(Xc2[end][3:3]) * Ac' * λp

        normal_cw = +x_2d_rotation(zeros(3)[3:3]) * Af' * λc
        R = [0 1; -1 0]
        tangent_pw = R * normal_pw
        tangent_cw = R * normal_cw

        j == 1 ? push!(C[length(contacts) + j], c + Xp2[end][1:2]) :
            push!(C[length(contacts) + j], c + Xc2[end][1:2])

        push!(Np[length(contacts) + j], normal_pw)
        push!(Nc[length(contacts) + j], normal_cw)
        push!(Tp[length(contacts) + j], tangent_pw)
        push!(Tc[length(contacts) + j], tangent_cw)
    end

end


scatter(iter)
plot!(hcat([s[solver.indices.primals] for s in solutions]...)', legend=false)
scatter(iter)
plot!(hcat([s[solver.indices.duals] for s in solutions]...)', legend=false)
scatter(iter)
plot!(hcat([s[solver.indices.slacks] for s in solutions]...)', legend=false)


################################################################################
# visualization
################################################################################
render(vis)
set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polytope!(vis, Ap, bp, name=:pbody, color=RGBA(0.2,0.2,0.2,0.7))
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
        set_2d_polytope!(vis, Xc2[i][1:2], Xc2[i][3:3], name=:cbody);
    end;
end;
MeshCat.setanimation!(vis, anim)
# open(vis)
# convert_frames_to_video_and_gif("single_level_hard_offset")
# convert_frames_to_video_and_gif("single_level_hard_tilted")
# convert_frames_to_video_and_gif("single_level_hard")

ex = solver.data.jacobian_variables_dense
plot(Gray.(abs.(ex)))
plot(Gray.(abs.(ex - ex')))
plot(Gray.(abs.(ex + ex')))
# plot(Gray.(1e3abs.(solver.data.jacobian_variables_dense)))

# scatter(solver.solution.all)
# scatter(solver.solution.primals)
# scatter(solver.solution.duals)
# scatter(solver.solution.slacks)

# plot(hcat(solutions...)', legend=false)
