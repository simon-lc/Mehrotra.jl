using Plots
using MeshCat
using Polyhedra
using GeometryBasics
using RobotVisualizer
using Quaternions
using StaticArrays


include("polyhedron.jl")
include("residual_2d_polytope.jl")
include("visuals.jl")
include("quaternion.jl")
include("rotate.jl")

vis = Visualizer()
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

Aa = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
ba = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]

Ab = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bb = 0.5*[
     1,
     1,
     1,
     1,
    ]
na = length(ba)
nb = length(bb)
d = 2

build_2d_polyhedron!(vis, Aa, ba, color=RGBA(0.2,0.2,0.2,0.6), name=:polya)
build_2d_polyhedron!(vis, Ab, bb, color=RGBA(0.8,0.8,0.8,0.6), name=:polyb)

xa2 = [2,3.0]
xb2 = [0,4.0]
qa2 = [+0.0]
qb2 = [-0.5]

parameters = pack_lp_parameters(xa2, qa2, xb2, qb2, Aa, ba, Ab, bb)

p1 = pack_lp_parameters(unpack_lp_parameters(parameters, na=na, nb=nb, d=d)...)


num_primals = d + 1
num_cone = na + nb

idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

sized_lp_residual(primals, duals, slacks, parameters) = lp_residual(
    primals, duals, slacks, parameters; na=na, nb=nb, d=d)

solver = Solver(
        sized_lp_residual,
        num_primals, num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Options228(),
        )

solver.options.compressed_search_direction = false
solver.options.max_iterations = 30
# solver.options.verbose = false
solve!(solver)


# pa = solver.solution.primals[1:d]
# l = solver.solution.primals[d .+ (1:1)]
# # pw is expressed in world's frame
# pw = xa2 + x_2d_rotation(qa2) * pa

pw = solver.solution.primals[1:d]
l = solver.solution.primals[d .+ (1:1)]

build_2d_polyhedron!(vis, Aa, ba, color=RGBA(0.2,0.2,0.2,0.6), name=:polya)
build_2d_polyhedron!(vis, Ab, bb, color=RGBA(0.8,0.8,0.8,0.6), name=:polyb)
build_2d_polyhedron!(vis, Aa, ba .+ l, color=RGBA(0.2,0.2,0.2,0.6), name=:poly_exta)
build_2d_polyhedron!(vis, Ab, bb .+ l, color=RGBA(0.8,0.8,0.8,0.6), name=:poly_extb)
set_2d_polyhedron!(vis, xa2, qa2, name=:polya)
set_2d_polyhedron!(vis, xb2, qb2, name=:polyb)
set_2d_polyhedron!(vis, xa2, qa2, name=:poly_exta)
set_2d_polyhedron!(vis, xb2, qb2, name=:poly_extb)
setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,pw...), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
