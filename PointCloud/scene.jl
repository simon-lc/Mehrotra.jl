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
    scene = Scene(resolution=resolution)
    cam3d!(scene)
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
    vis.screen[1] = display(vis.scene[:root], visible=visible)
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
    (vis.screen[1] == nothing) && open(vis, visible=false)
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



vis = GLVisualizer1190(resolution=(100, 1000))
open(vis, visible=false)

vis.scene[:root].px_area.val.widths


floor1 = HyperRectangle(Vec(-2,-2,-0.1), Vec(4, 4, 0.1))
floor2 = HyperRectangle(Vec(-1,-1,-0.2), Vec(2, 2, 0.25))
object1 = HyperRectangle(Vec(0,0,0), Vec(0.1, 0.4, 1))
object2 = HyperRectangle(Vec(0,0,0), Vec(0.2, 0.1, 2.0))

setobject!(vis, :root, :floor1, floor1, color=RGBA(0,1,1,0.1))
setobject!(vis, :root, :floor2, floor2, color=RGBA(0.3,0.3,1,0.1))
setobject!(vis, :floor1, :object1, object1, color=RGBA(0,0,1,1))
setobject!(vis, :object1, :object2, object2, color=RGBA(1,0,0,1))

# settransform!(vis, :object1, [0,0,1.0], Quaternion(0,0,0,1.0))
# settransform!(vis, :object1, [0,0,1.0], Quaternion(sqrt(2)/2,0,0,sqrt(2)/2))
# settransform!(vis, :object1, [0,0,1.0], Quaternion(1,0,0,0.0))

depth_color = depth_buffer(vis)
depth_buffer!(depth_color, vis)
# @benchmark $depth_buffer($vis)
# @benchmark $depth_buffer!($depth_color, $vis)

# Plots.spy(10*rotl90(depth_color))
Plots.plot(Gray.(15*(1 .- rotl90(depth_color))))


function set_camera!(vis::GLVisualizer1190)

    return nothing
end

function set_resolution!(vis::GLVisualizer1190)

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
screen = display(scene)
GLMakie.depthbuffer(screen)





GLMakie.ShaderAbstractions.switch_context!(screen.glscreen)
GLMakie.render_frame(screen, resize_buffers=false) # let it render
GLMakie.glFinish() # block until opengl is done rendering
source = screen.framebuffer.buffers[:depth]
depth = Matrix{Float32}(undef, size(source))
GLMakie.GLAbstraction.bind(source)
GLMakie.GLAbstraction.glGetTexImage(source.texturetype, 0, GLMakie.GL_DEPTH_COMPONENT, GLMakie.GL_FLOAT, depth)
GLMakie.GLAbstraction.bind(source, 0)


depthbuffer!(screen, depth)



# GLMakie.translate!(vis.scene[:floor1], 0,0,0.2)
# GLMakie.translate!(vis.scene[:object1], 0,0,0.2)
# GLMakie.translate!(vis.scene[:object2], 0,0,0.2)
#
# GLMakie.rotate!(vis.scene[:floor1], Vec3f(0, 0, 1), 0.2)
# GLMakie.rotate!(vis.scene[:object1], Vec3f(0, 0, 1), 0.2)
# GLMakie.rotate!(vis.scene[:object2], Vec3f(0, 0, 1), 0.2)
# GLMakie.rotate!(vis.scene[:object2], q)
# q = Quaternion(normalize([0,1,1,1.0])...)


child = Transformation(vis.scene)
mesh!(scene, obj; color=:blue, transformation=child)


set_object!(vis, object, name)


# load the model file
m = load(assetpath("lego_figure_" * name * ".stl"))
# look up color
color = colors[split(name, "_")[1]]
# Create a child transformation from the parent
child = Transformation(parent)
# get the transformation of the parent
ptrans = Makie.transformation(parent)
# get the origin if available
origin = get(origins, name, nothing)
# center the mesh to its origin, if we have one
if !isnothing(origin)
    centered = m.position .- origin
    m = GeometryBasics.Mesh(meta(centered; normals=m.normals), faces(m))
    translate!(child, origin)
else
    # if we don't have an origin, we need to correct for the parents translation
    translate!(child, -ptrans.translation[])
end
# plot the part with transformation & color
mesh!(scene, obj; color=:blue, transformation=child)




function plot_part!(scene, parent, name::String)
    # load the model file
    m = load(assetpath("lego_figure_" * name * ".stl"))
    # look up color
    color = colors[split(name, "_")[1]]
    # Create a child transformation from the parent
    child = Transformation(parent)
    # get the transformation of the parent
    ptrans = Makie.transformation(parent)
    # get the origin if available
    origin = get(origins, name, nothing)
    # center the mesh to its origin, if we have one
    if !isnothing(origin)
        centered = m.position .- origin
        m = GeometryBasics.Mesh(meta(centered; normals=m.normals), faces(m))
        translate!(child, origin)
    else
        # if we don't have an origin, we need to correct for the parents translation
        translate!(child, -ptrans.translation[])
    end
    # plot the part with transformation & color
    return mesh!(scene, m; color=color, transformation=child)
end


function plot_lego_figure(s, floor=true)
    # Plot hierarchical mesh and put all parts into a dictionary
    figure = Dict()
    figure["torso"] = plot_part!(s, s, "torso")
    figure["head"] = plot_part!(s, figure["torso"], "head")
    figure["eyes_mouth"] = plot_part!(s, figure["head"], "eyes_mouth")
    figure["arm_right"] = plot_part!(s, figure["torso"], "arm_right")
    figure["hand_right"] = plot_part!(s, figure["arm_right"], "hand_right")
    figure["arm_left"] = plot_part!(s, figure["torso"], "arm_left")
    figure["hand_left"] = plot_part!(s, figure["arm_left"], "hand_left")
    figure["belt"] = plot_part!(s, figure["torso"], "belt")
    figure["leg_right"] = plot_part!(s, figure["belt"], "leg_right")
    figure["leg_left"] = plot_part!(s, figure["belt"], "leg_left")

    # lift the little guy up
    translate!(figure["torso"], 0, 0, 20)
    # add some floor
    floor && mesh!(s, Rect3f(Vec3f(-400, -400, -2), Vec3f(800, 800, 2)), color=:white)
    return figure
end


scene = Scene()
scale!(scene, 0.5, 0.5, 0.5)
GLMakie.rotate!(scene, Vec3f(1, 0, 0), 0.5) # 0.5 rad around the y axis
scene

GLMakie.translate!(sphere_plot, Vec3f(0, 0, 1))
scene


parent_scene = Scene()
cam3d!(parent_scene)

# One can set the camera lookat and eyeposition, by getting the camera controls and using `update_cam!`
camc = cameracontrols(parent_scene)
update_cam!(parent_scene, camc, Vec3f(0, 8, 0), Vec3f(4.0, 0, 0))

s1 = Scene(parent_scene, camera=parent_scene.camera)
mesh!(s1, Rect3f(Vec3f(0, -0.1, -0.1), Vec3f(5, 0.2, 0.2)))
s2 = Scene(s1, camera=parent_scene.camera)
mesh!(s2, Rect3f(Vec3f(0, -0.1, -0.1), Vec3f(5, 0.2, 0.2)), color=:red)
GLMakie.translate!(s2, 5, 0, 0)
s3 = Scene(s2, camera=parent_scene.camera)
mesh!(s3, Rect3f(Vec3f(-0.2), Vec3f(0.4)), color=:blue)
GLMakie.translate!(s3, 5, 0, 0)
parent_scene





# shapes
pyr = Pyramid(GeometryBasics.Point3f(0), 1.0f0, 1.0f0)
rectmesh = Rect3(GeometryBasics.Point3f(-0.5), GeometryBasics.Vec3f(1))
sphere = Sphere(GeometryBasics.Point3f(-0.5), 1)
Cone(; quality=10) = merge([
    Makie._circle(GeometryBasics.Point3f(0), 0.5f0, GeometryBasics.Vec3f(0, 0, -1), quality),
    Makie._mantle(GeometryBasics.Point3f(0), GeometryBasics.Point3f(0, 0, 1), 0.5f0, 0.0f0, quality)])
cone = Cone()

# meshes
brain = load(assetpath("brain.stl"))
matball_outer = load(assetpath("matball_outer.obj"))

function plotmat()
    ambient = GeometryBasics.Vec3f(0.8, 0.8, 0.8)
    shading = true
    fig = Figure(resolution=(1200, 900))
    axs = [LScene(fig[i, j]; show_axis=false)
           for j in 1:1, i in 1:1]
    mesh!(axs[1], brain; shading, ambient)
    GLMakie.rotate!(axs[1].scene, 2.35)
    center!(axs[1].scene)
    zoom!(axs[1].scene, cameracontrols(axs[1].scene), 0.75)
    fig
end
fig = with_theme(plotmat, theme_dark())

# x = scatter(1:4)
screen = display(fig)

function get_depth(screen)
    depth_color = GLMakie.depthbuffer(screen)
end

function get_depth!(depth_color, screen)
    depth_color .= GLMakie.depthbuffer(screen)
end

depth_color = get_depth(screen)
get_depth!(depth_color, screen)
Plots.plot(Gray.(10(1 .- depth_color)))
# @benchmark $get_depth($screen)
# @benchmark $get_depth!($depth_color, $screen)

set_window_config!(;
    # renderloop = renderloop,
    vsync = true,
    # framerate = 30.0,
    # float = false,
    # pause_rendering = false,
    # focus_on_show = false,
    decorated = true,
    title = "Makie"
)
screen = display(fig)
# # Look at result:








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
    # theme_kw...
)


GLMakie.activate!()
scene = Scene(backgroundcolor=:gray)
subwindow = Scene(scene, px_area=Rect(100, 100, 200, 200), clear=true, backgroundcolor=:white)
scene

cam3d!(subwindow)
meshscatter!(subwindow, rand(Point3f, 10), color=:gray)
center!(subwindow)
scene

subwindow.clear = false
relative_space = Makie.camrelative(subwindow)
# this draws a line at the scene window boundary
lines!(relative_space, Rect(0, 0, 1, 1))
scene

campixel!(scene)
w, h = size(scene) # get the size of the scene in pixels
# this draws a line at the scene window boundary
image!(scene, [sin(i/w) + cos(j/h) for i in 1:w, j in 1:h])
scene


scene = Scene(backgroundcolor=:gray)
lines!(scene, Rect2f(-1, -1, 2, 2), linewidth=5, color=:black)
scene

cam = Makie.camera(scene) # this is how to access the scenes camera

cam.projection[] = Makie.orthographicprojection(-3f0, 5f0, -3f0, 5f0, -100f0, 100f0)
scene

w, h = size(scene)
nearplane = 0.1f0
farplane = 100f0
aspect = Float32(w / h)
cam.projection[] = Makie.perspectiveprojection(45f0, aspect, nearplane, farplane)
# Now, we also need to change the view matrix
# to "put" the camera into some place.
eyeposition = Vec3f(10)
lookat = Vec3f(0)
upvector = Vec3f(0, 0, 1)
cam.view[] = Makie.lookat(eyeposition, lookat, upvector)
scene

screen = display(scene)





scene = Scene()
cam3d!(scene)
display(scene)

sphere_plot = mesh!(scene, Sphere(Point3f(0, 0, 0), 0.1), color=:black)
sphere_plot = mesh!(scene, Sphere(Point3f(1, 0, 0), 0.1), color=:red)
sphere_plot = mesh!(scene, Sphere(Point3f(0, 1, 0), 0.1), color=:green)
sphere_plot = mesh!(scene, Sphere(Point3f(0, 0, 1), 0.1), color=:blue)
x = 10
y = 10
z = 0.1
obj = HyperRectangle(Vec(-x/2, -y/2, -z), Vec(x, y, z))
sphere_plot = mesh!(scene, obj, color=RGBA(0.4, 0.4, 0.4, 1.0))


parent_scene = Scene()
cam3d!(parent_scene)

# One can set the camera lookat and eyeposition, by getting the camera controls and using `update_cam!`
camc = cameracontrols(parent_scene)
update_cam!(parent_scene, camc, Vec3f(0, 8, 0), Vec3f(4.0, 0, 0))

s1 = Scene(parent_scene)#, camera=parent_scene.camera)
mesh!(s1, Rect3f(Vec3f(0, -0.1, -0.1), Vec3f(5, 0.2, 0.2)))
display(parent_scene)

s2 = Scene(s1)#, camera=parent_scene.camera)
mesh!(s2, Rect3f(Vec3f(0, -0.1, -0.1), Vec3f(5, 0.2, 0.2)), color=:red)
GLMakie.translate!(s2, 5, 0, 0)
display(parent_scene)

s3 = Scene(s2)#, camera=parent_scene.camera)
mesh!(s3, Rect3f(Vec3f(-0.2), Vec3f(0.4)), color=:blue)
GLMakie.translate!(s3, 5, 0, 0)
display(parent_scene)


# Now, rotate the "joints"
GLMakie.rotate!(s2, Vec3f(0, 1, 0), 0.5)
GLMakie.rotate!(s3, Vec3f(1, 0, 0), 0.5)


display(parent_scene)
