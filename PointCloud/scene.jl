using GLMakie
using FileIO
using Downloads
using JSON
using Makie.GeometryBasics: Pyramid
using GeometryBasics
using Colors
using LinearAlgebra
using Plots
using Makie
using Random
using Graphs
using Quaternions
using RobotVisualizer
using BenchmarkTools

struct GLVisualizer1190
    scene::Dict{Symbol,Any}
    trans::Dict{Symbol,Any}
    names::Vector{Symbol}
    graph::SimpleDiGraph
    screen::Vector
end

function GLVisualizer1190(; resolution=(800,600))
    scene = Scene(
        # clear everything behind scene
        clear = true,
        # the camera struct of the scene.
        visible = true,
        resolution=resolution)
    cam3d_cad!(scene)
    scene = Dict{Symbol,Any}(:root => scene)
    trans = Dict{Symbol,Any}()
    names = [:root]
    graph = SimpleDiGraph()
    add_vertex!(graph)
    screen = Vector{Any}()
    push!(screen, nothing)
    return GLVisualizer1190(scene, trans, names, graph, screen)
end

function Base.open(vis::GLVisualizer1190; visible::Bool=true)
    vis.screen[1] = display(vis.scene[:root])#, visible=visible)
end

function setobject!(vis::GLVisualizer1190, parent::Symbol, name::Symbol, object;
        color=RGBA(0.3, 0.3, 0.3, 0.7))

    parent_scene = vis.scene[parent]
    child_scene = Scene(parent_scene, camera=vis.scene[:root].camera)
    vis.scene[name] = child_scene

    mesh!(child_scene, object; color=color)
    vis.trans[name] = Transformation(parent_scene)
    push!(vis.names, name)
    add_vertex!(vis.graph)

    child_id = length(vis.names)
    parent_id = findfirst(x -> x==parent, vis.names)
    add_edge!(vis.graph, parent_id, child_id)
    return nothing
end

function settransform!(vis::GLVisualizer1190, name::Symbol, x, q)
    set_translation!(vis, name, x)
    set_rotation!(vis, name, q)
    return nothing
end

function set_translation!(vis::GLVisualizer1190, name::Symbol, x)
    GLMakie.translate!(vis.scene[name], x...)
    return nothing
end

function set_rotation!(vis::GLVisualizer1190, name::Symbol, q)
    GLMakie.rotate!(vis.scene[name], q)
    return nothing
end

function depth_buffer(vis::GLVisualizer1190)
    depth = depthbuffer!(vis.screen[1])
end

function depth_buffer!(depth::Matrix, vis::GLVisualizer1190)
    (vis.screen[1] == nothing) && open(vis, visible=true)
    depthbuffer!(vis.screen[1], depth)
end

function depthbuffer!(screen::GLMakie.Screen, depth=Matrix{Float32}(undef, size(screen.framebuffer.buffers[:depth])))
    GLMakie.ShaderAbstractions.switch_context!(screen.glscreen)
    GLMakie.render_frame(screen, resize_buffers=false) # let it render
    GLMakie.glFinish() # block until opengl is done rendering
    source = screen.framebuffer.buffers[:depth]
    @assert size(source) == size(depth)
    GLMakie.GLAbstraction.bind(source)
    GLMakie.GLAbstraction.glGetTexImage(source.texturetype, 0, GLMakie.GL_DEPTH_COMPONENT, GLMakie.GL_FLOAT, depth)
    GLMakie.GLAbstraction.bind(source, 0)
    return depth
end

vis = GLVisualizer1190(resolution=(150, 150))
open(vis, visible=true)

floor1 = HyperRectangle(Vec(-2,-2,-0.1), Vec(4, 4, 0.1))
floor2 = HyperRectangle(Vec(-1,-1,-0.2), Vec(2, 2, 0.25))
object1 = HyperRectangle(Vec(0,0,0), Vec(0.1, 0.4, 1))
object2 = HyperRectangle(Vec(0,0,0), Vec(0.2, 0.1, 2.0))

setobject!(vis, :root, :floor1, floor1, color=RGBA(0,1,1,0.1))
setobject!(vis, :root, :floor2, floor2, color=RGBA(0.3,0.3,1,0.1))
setobject!(vis, :floor1, :object1, object1, color=RGBA(0,0,1,1))
setobject!(vis, :object1, :object2, object2, color=RGBA(1,0,0,1))


eyeposition = [0, 0, 4]
lookat = [0, 0, 0]
up = [0, 1, 0]
update_cam!(vis.scene[:root], eyeposition, lookat, up)
# for (key,val) in vis.scene
#     @show key
#     update_cam!(vis.scene[key], eyeposition, lookat, up)
# end


# settransform!(vis, :object1, [0,0,1.0], Quaternion(0,0,0,1.0))
# settransform!(vis, :object1, [0,0,1.0], Quaternion(sqrt(2)/2,0,0,sqrt(2)/2))
# settransform!(vis, :object1, [0,0,1.0], Quaternion(1,0,0,0.0))

depth_color = depth_buffer(vis)
depth_buffer!(depth_color, vis)
# @benchmark $depth_buffer($vis)
# @benchmark $depth_buffer!($depth_color, $vis)

# Plots.spy(10*rotl90(depth_color))
Plots.plot(Gray.(15*(1 .- rotl90(depth_color))))
depth_color
maximum(depth_color)
minimum(depth_color)

function set_camera!(vis::GLVisualizer1190)

    return nothing
end

scene = Scene(;
    # clear everything behind scene
    clear = true,
    # the camera struct of the scene.
    visible = true,
    # ssao and light are explained in more detail in `Documetation/Lighting`
    ssao = Makie.SSAO(),
    # Creates lights from theme, which right now defaults to `
    # set_theme!(lightposition=:eyeposition, ambient=RGBf(0.5, 0.5, 0.5))`
    lights = Makie.automatic,
    backgroundcolor = :gray,
    resolution = (500, 500)
    # gets filled in with the currently set global theme
)
screen = display(scene, visble=true)
screen = display(scene)
screen = display(scene, visble=true)
screen = display(scene, visble=true)
screen = display(scene, visble=true)
for i = 1:100
    screen = display(scene)
end
GLMakie.depthbuffer(screen)
vis.scene[:root]
