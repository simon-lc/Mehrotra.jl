using Plots

d = 2.0
W = 1/3^2

f1(x, d) = 0.5 * (x - d) * (x - d)
f2(x) = - 0.5 * x * W * x
# g2(x) = - 0.25 * x * sqrt(W) * x
# grad(x, d) = (x - d) - sqrt(W) * x

N = 1000
X = range(-8, 8, length=N)
plt = plot(
    xlims=(-6,6),
    ylims=(-4,5),
    aspectratio=1,
    )
plot!(plt, X, zeros(N), color=:black, linewidth=1)
plot!(plt, X, f1.(X,d), color=:blue, linewidth=1)
plot!(plt, X, f2.(X), color=:red, linewidth=1)
plot!(plt, X, f1.(X,d) + f2.(X), color=:green, linewidth=3)
plot!(plt, X, f1.(X,d) + f2.(X) .+ 1/d .- 1/d .* X, color=:green, linewidth=3)
# plot!(plt, X, g2.(X), color=:red, linewidth=2)
# plot!(plt, X, f1.(X,d) + g2.(X), color=:green, linewidth=4)
# plot!(plt, X, grad.(X,d), color=:green, linewidth=4)
plot!([-3, -3], [-10, +10], color=:red)
plot!([+3, +3], [-10, +10], color=:red)

plot!([-1+d, -1+d], [-0, +1], color=:black, linewidth=2.0)
plot!([+1+d, +1+d], [-0, +1], color=:black, linewidth=2.0)


# 3D plot
using RobotVisualizer
using Meshing
using GeometryBasics

vis = Visualizer()
render(vis)

function RobotVisualizer.set_surface!(vis::Visualizer, f::Any;
    xlims=[-20.0, 20.0],
    ylims=[-20.0, 20.0],
    zlims=[-2.0, 4.0],
    color=RGBA(1.0, 1.0, 1.0, 1.0),
    wireframe=false,
    n::Int=200)
    mesh = GeometryBasics.Mesh(f,
        MeshCat.HyperRectangle(
            MeshCat.Vec(xlims[1], ylims[1], zlims[1]),
            MeshCat.Vec(xlims[2] - xlims[1], ylims[2] - ylims[1], zlims[2] - zlims[1])),
        Meshing.MarchingCubes(), samples=(n, n, Int(floor(n / 8))))
    setobject!(vis["surface"], mesh, MeshPhongMaterial(color=color, wireframe=wireframe))
    return nothing
end

w1 = 1/1
w2 = 1/16
W = [w1 0; 0  w2]
b1 = 1/1
b2 = 1/1
B = [b1 0; 0 b2]

p = [0.0, 3.0]
fW(x) = (x[1:2]' * W * x[1:2]) - 100x[3] - 1
gW(x) = -0.5*(x[1:2]' * sqrt(W) * x[1:2]) - x[3]

function fB(x, p)
    xy = x[1:2] - p
    return (xy' * B * xy) - 100x[3] - 1
end
function gB(x, p)
    xy = x[1:2] - p
    return 0.5*(xy' * B * xy) - x[3]
end

set_surface!(vis[:world], fW, n=50, wireframe=true,
    xlims=[-10,10],
    ylims=[-10,10],
    zlims=[-1,+0],
    color=RGBA(1,1,1,0.2))
set_surface!(vis[:body], x->fB(x,p), n=50, wireframe=true,
    xlims=p[1] .+ [-3,3],
    ylims=p[2] .+ [-3,3],
    zlims=[-1,+0],
    color=RGBA(1,0,0,0.5))


set_surface!(vis[:world_cost], gW, n=50, wireframe=false,
    xlims=[-10,10],
    ylims=[-10,10],
    zlims=[-2,+0],
    color=RGBA(1,1,1,0.2))
set_surface!(vis[:body_cost], x->gB(x,p), n=50, wireframe=false,
    xlims=p[1] .+ [-3,3],
    ylims=p[2] .+ [-3,3],
    zlims=[-0,0.5],
    color=RGBA(1,0,0,0.5))
set_surface!(vis[:body_extend], x->gB(x,p), n=50, wireframe=false,
    xlims=p[1] .+ [-3,3],
    ylims=p[2] .+ [-3,3],
    zlims=[-0,1.0],
    color=RGBA(1,0,0,0.2))
set_surface!(vis[:sum_cost], x -> gW(x) + gB(x,p), n=50, wireframe=false,
    xlims=[-10,10],
    ylims=[-10,10],
    zlims=[-2,-0.0],
    color=RGBA(1,1,0,0.2))
