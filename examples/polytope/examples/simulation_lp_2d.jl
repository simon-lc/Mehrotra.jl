using Plots
using MeshCat
using Polyhedra
using GeometryBasics
using RobotVisualizer
using Quaternions
using StaticArrays

include("../polyhedron.jl")
include("../contact_model/lp_simulator_2d.jl")
include("../visuals.jl")
include("../quaternion.jl")
include("../rotate.jl")

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

xa2 = [1,3.0]
xb2 = [0,4.0]
qa2 = [+0.5]
qb2 = [-0.5]

va15 = [0,0.0]
ωa15 = [+1.0]
vb15 = [0,0.0]
ωb15 = [-1.0]

u = zeros(6)
timestep = 0.01
mass = 1.0
inertia = 0.1
gravity = -0.0*9.81

parameters = pack_lp_simulator_parameters(
    xa2, qa2, xb2, qb2,
    va15, ωa15, vb15, ωb15,
    u, timestep, mass, inertia, gravity,
    Aa, ba, Ab, bb)


num_primals = d+1 + 2d+2
num_cone = na + nb + 1

idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

sized_lp_simulator_residual(primals, duals, slacks, parameters) = lp_simulator_residual(
    primals, duals, slacks, parameters; na=na, nb=nb, d=d)

solver = Solver(
        sized_lp_simulator_residual,
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

J0 = Matrix(solver.data.jacobian_variables)
plot(Gray.(abs.(J0)))
plot(Gray.(abs.(J0 - J0')))


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




################################################################################
# simulation
################################################################################
xa2 = [0.4,3.0]
xb2 = [0,4.0]
qa2 = [+0.5]
qb2 = [-0.5]

va15 = [0,0.0]
ωa15 = [+1.0]
vb15 = [0,0.0]
ωb15 = [-1.0]
H = 400
U = [zeros(6) for i=1:H]
Pw, Xa, Qa, Xb, Qb, Va, Ωa, Vb, Ωb, iterations = simulate_lp(solver, xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U;
        timestep=timestep,
        mass=mass,
        inertia=inertia,
        friction_coefficient=0.2,
        gravity=gravity)


################################################################################
# visualization
################################################################################
build_2d_polyhedron!(vis, Aa, ba, name=:polya, color=RGBA(0.2,0.2,0.2,0.7))
build_2d_polyhedron!(vis, Ab, bb, name=:polyb, color=RGBA(0.9,0.9,0.9,0.7))

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

anim = MeshCat.Animation(Int(floor(1/timestep)))
for i = 1:H
    atframe(anim, i) do
        set_2d_polyhedron!(vis, Xa[i], Qa[i], name=:polya)
        set_2d_polyhedron!(vis, Xb[i], Qb[i], name=:polyb)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Pw[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
