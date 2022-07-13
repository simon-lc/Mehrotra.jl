using Mehrotra
using Random
using RobotVisualizer
using Colors
using Polyhedra
using StaticArrays
using Quaternions

include("../polytope/contact_model/lp_2d.jl");
include("../polytope/polytope.jl");
include("../polytope/quaternion.jl");
include("../polytope/rotate.jl");
include("../polytope/visuals.jl");


# parameters
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2)
bp = 0.5*[
    +1,
    +1,
    +1,
     1,
     # 2,
    ] - 0.0*ones(4)
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.00ones(4,2)
bc = 0.5*[
     1,
     1,
     1,
     1,
    ] + 0.0*rand(4)

solver = lp_contact_solver(Ap, bp, Ac, bc; d=2,
    options=Options(
        verbose=false,
        complementarity_tolerance=3e-3,
        residual_tolerance=1e-6,
        differentiate=true,
        compressed_search_direction=false,
        sparse_solver=true,
        ));

d = 2
xp = [40, 40, 0.0]
xc = [00, 00, 0.0]
np = length(bp)
nc = length(bc)

solver.parameters
dimensions = solver.dimensions
solver.parameters[1:2d+2] .= [xp; xc]
solve!(solver)
solver.solution.primals
zp = solver.solution.duals[1:np]
zc = solver.solution.duals[np .+ (1:nc)]

x_2d_rotation(xp[3:3]) * Ap' * zp
x_2d_rotation(xc[3:3]) * Ac' * zc


function contact_information(solver::Solver, xp, xc; np=4, nc=4)
    solver.parameters[1:6] .= [xp; xc]
    solve!(solver)
    solution = solver.solution

    xw = solution.primals[1:2] + (xp[1:2] + xc[1:2]) ./ 2 
    d = solution.primals[3:3]
    zp = solution.duals[1:np]

    nw = -x_2d_rotation(xp[3:3]) * Ap' * zp
    R = [0 1; -1 0]
    tw = R * nw
    return d, xw, nw, tw
end

function set_contact_information!(vis::Visualizer, d, xw, nw, tw)
    vis[]
    return nothing
end

#########################################################################
# generate contact info
#########################################################################
H = 200
θ = range(0, 6π, length=H)
rad = 4.0
Xc = [[0, 4, 0.0] for i=1:H]
Xp = [[1, 5, 0.0] for i=1:H]
# Xp = [[-1+2i/H, 5, 0.0] for i=1:H]
Xp = [[-0.75+1.5i/H, 4.5+sqrt(2)/2, π/4] for i=1:H]
# Xp = [[ i/H*rad*cos(t), i/H*rad*sin(t)+4, 0.0] for (i,t) in enumerate(θ)]
D = []
Xw = []
Nw = []
Tw = []
for i = 1:H
    d, xw, nw, tw = contact_information(solver, Xp[i], Xc[i])
    push!(D, d)
    push!(Xw, xw)
    push!(Nw, nw)
    push!(Tw, tw)
end

using Plots
# plot(hcat(Nw...)')
# plot(hcat(Tw...)')
plot(norm.(Nw))

#########################################################################
# visualize
#########################################################################
# vis = Visualizer()
render(vis)

set_floor!(vis)
set_background!(vis)
set_light!(vis)

build_2d_polytope!(vis, Ap, bp, name=:parent, color=RGBA(0.2, 0.2, 0.2, 0.9))
build_2d_polytope!(vis, Ac, bc, name=:child, color=RGBA(0.9, 0.9, 0.9, 0.9))

setobject!(vis[:contact],
    HyperSphere(GeometryBasics.Point(0,0,0.), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

build_rope(vis; N=1, color=Colors.RGBA(0,0,0,1),
    rope_type=:cylinder, rope_radius=0.04, name=:normal)

build_rope(vis; N=1, color=Colors.RGBA(1,0,0,1),
    rope_type=:cylinder, rope_radius=0.04, name=:tangent)

timestep = 0.02
anim = MeshCat.Animation(Int(floor(1/timestep)))

for i = 1:H
    atframe(anim, i) do
        set_straight_rope(vis, [0; Xw[i]], [0; Xw[i]+Nw[i]]; N=1, name=:normal)
        set_straight_rope(vis, [0; Xw[i]], [0; Xw[i]+Tw[i]]; N=1, name=:tangent)
        set_2d_polytope!(vis, Xp[i][1:2], Xp[i][3:3], name=:parent)
        set_2d_polytope!(vis, Xc[i][1:2], Xc[i][3:3], name=:child)
        settransform!(vis[:contact], MeshCat.Translation(SVector{3}(0, Xw[i]...)))
    end
end
MeshCat.setanimation!(vis, anim)
# open(vis)
# convert_frames_to_video_and_gif("contact_normal_vertex_sliding")