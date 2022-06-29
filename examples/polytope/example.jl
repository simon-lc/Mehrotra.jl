using Plots
using MeshCat
using Polyhedra
using GeometryBasics
using RobotVisualizer
using Quaternions
using StaticArrays


include("polyhedron.jl")
# include("signed_distance.jl")
# include("residual.jl")
include("residual_qp.jl")
include("visuals.jl")
include("quaternion.jl")
include("rotate.jl")


vis = Visualizer()
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

# ContactPoint pt;
# pt.world_normal_on_b = normal_on_b;
# pt.world_point_on_a = point_a_world;
# pt.world_point_on_b = point_b_world;
# pt.distance = distance;

################################################################################
# contact
################################################################################
function contact_solver(parameters; n1=0, n2=0, d=0,
        options::Options228=Options228(max_iterations=30, verbose=true))

    num_primals = 2d
    num_cone = n1 + n2

    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    sized_contact_residual(primals, duals, slacks, parameters) = contact_residual(
        primals, duals, slacks, parameters; n1=n1, n2=n2, d=d)

    solver = Solver(sized_contact_residual, num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=options,

        )
    return solver
end

function set_poses!(solver::Solver228, x1, q1, x2, q2)
    d = length(x1)
    off = 0
    solver.parameters[off .+ (1:d)] .= x1; off += d
    solver.parameters[off .+ (1:d)] .= q1; off += 3
    solver.parameters[off .+ (1:d)] .= x2; off += d
    solver.parameters[off .+ (1:d)] .= q2; off += 3
    return nothing
end

function contact_distance(solver, x1, q1, x2, q2)
    d = length(x1)
    set_poses!(solver, x1, q1, x2, q2)
    solve!(solver)

    y = solver.solution.primals
    # y1 is expressed in body1's frame
    y1 = y[1:d]
    # y2 is expressed in body2's frame
    y2 = y[d .+ (1:d)]
    # yw is expressed in world frame
    y1w = x1 + z_rotation(q1) * y1
    y2w = x2 + z_rotation(q2) * y2
    return norm(y1w - y2w), y1w, y2w
end

# function contact_jacobian(x1, q1, x2, q2, A1, b1, A2, b2, δ)
#     return distance_jacobian_inputs, tangential_velocities, rotational_velocity
# end




A1 = [
     1.0  0.0  0.0;
     0.0  1.0  0.0;
     0.0  0.0  1.0;
    -1.0  0.0  0.0;
     0.0 -1.0  0.0;
     0.0  0.0 -1.0;
    ] .- 0.10ones(6,3)
b1 = 0.5*[
    +1,
    +1,
    +1,
    +1,
     2,
     1,
    ]

A2 = [
     1.0  0.0  0.0;
     0.0  1.0  0.0;
     0.0  0.0  1.0;
    -1.0  0.0  0.0;
     0.0 -1.0  0.0;
     0.0  0.0 -1.0;
    ] .+ 0.10ones(6,3)
b2 = 0.5*[
     1,
     1,
     1,
     1,
     1,
     3,
    ]
δ = 1e2

build_polyhedron!(vis, A1, b1, color=RGBA(0.2,0.2,0.2,0.6), name=:poly1)
build_polyhedron!(vis, A2, b2, color=RGBA(0.8,0.8,0.8,0.6), name=:poly2)

x1 = [2,1,1.0]
x2 = [0,0,3.0]
# q1 = Quaternion(1,0,0,0.0)
# q2 = Quaternion(1,0,0,0.0)
q1 = [0,0,0.5]
q2 = [0,0,-0.5]
# q1 = [0,0,0.0]
# q2 = [0,0,0.0]

set_polyhedron!(vis, x1, q1, name=:poly1)
set_polyhedron!(vis, x2, q2, name=:poly2)


parameters = pack_contact_parameters(x1, q1, x2, q2, A1, b1, A2, b2, δ)
solver = contact_solver(parameters, n1=size(A1)[1], n2=size(A2)[1], d=size(A1)[2])
distance, y1w, y2w = contact_distance(solver, x1, q1, x2, q2)


solver.data
solver.dimensions

solver.solution
A1*y1 - b1
A2*y2 - b2

setobject!(vis[:contact1],
    HyperSphere(GeometryBasics.Point(y1w...), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

setobject!(vis[:contact2],
    HyperSphere(GeometryBasics.Point(y2w...), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))







open(vis)







#
# primals = rand(6)
# duals = rand(12)
# slacks = rand(12)
# parameters = rand(61)
# contact_residual(primals, duals, slacks, parameters)
#
