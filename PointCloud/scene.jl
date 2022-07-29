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

struct GLVisualizer1220
    scene::Dict{Symbol,Any}
    trans::Dict{Symbol,Any}
    names::Vector{Symbol}
    graph::SimpleDiGraph
    screen::Vector
	camera::Camera3D
end

function GLVisualizer1220(; resolution=(800,600))
    scene = Scene(
        # clear everything behind scene
        clear = true,
        # the camera struct of the scene.
        visible = true,
        resolution=resolution)

	camera = cam3d_cad!(scene)
    scene = Dict{Symbol,Any}(:root => scene)
    trans = Dict{Symbol,Any}()
    names = [:root]
    graph = SimpleDiGraph()
    add_vertex!(graph)
	# screen
	screen = Vector{Any}()
    push!(screen, nothing)
    return GLVisualizer1220(scene, trans, names, graph, screen, camera)
end

function Base.open(vis::GLVisualizer1220; visible::Bool=true)
    vis.screen[1] = display(vis.scene[:root])#, visible=visible)
end

function setobject!(vis::GLVisualizer1220, parent::Symbol, name::Symbol, object;
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

function settransform!(vis::GLVisualizer1220, name::Symbol, x, q)
    set_translation!(vis, name, x)
    set_rotation!(vis, name, q)
    return nothing
end

function set_translation!(vis::GLVisualizer1220, name::Symbol, x)
    GLMakie.translate!(vis.scene[name], x...)
    return nothing
end

function set_rotation!(vis::GLVisualizer1220, name::Symbol, q)
    GLMakie.rotate!(vis.scene[name], q)
    return nothing
end

function depth_buffer(vis::GLVisualizer1220)
    depth = depthbuffer!(vis.screen[1])
end

function depth_buffer!(depth::Matrix, vis::GLVisualizer1220)
    (vis.screen[1] == nothing) && open(vis, visible=true)
    depthbuffer!(vis.screen[1], depth)
	# depth buffer return a value between [0,1] correspoding to distances between [camera.near, camera.far]

	# rescaling
	near_i = 1 / vis.camera.near.val
	far_i = 1 / vis.camera.far.val
	@show maximum(depth_color)
	@show minimum(depth_color)
	depth .= 1 ./ (near_i .+ depth .* (far_i - near_i))
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

function set_camera!(vis::GLVisualizer1220;
        eyeposition=[1,1,1.0],
        lookat=[0,0,0.0],
        up=[0,0,1.0],
		near=0.1,
		far=100.0,
		zoom=1.0)

	camera = vis.camera

	camera.lookat[] = Vec3f(lookat)
	camera.eyeposition[] = Vec3f(eyeposition)
	camera.upvector[] = Vec3f(up)
	camera.near[] = near
	camera.far[] = far
	camera.zoom_mult[] = zoom

    update_cam!(vis.scene[:root], camera)
    return nothing
end

"""
    set_floor!(vis; x, y, z, origin, normal, color)
    adds floor to visualization
    vis::Visualizer
    x: lateral position
    y: longitudinal position
    z: vertical position
	origin:: position of the center of the floor
    normal:: unit vector indicating the normal to the floor
    color: RGBA
"""
function RobotVisualizer.set_floor!(vis::GLVisualizer1220;
	    x=20.0,
	    y=20.0,
	    z=0.1,
	    origin=[0,0,0.0],
		normal=[0,0,1.0],
	    color=RGBA(0.5,0.5,0.5,1.0),
	    axis::Bool=false,
	    grid::Bool=false)
	obj = HyperRectangle(Vec(-x/2, -y/2, -z), Vec(x, y, z))
	setobject!(vis, :root, :floor, obj, color=color)

	p = origin
	q = RobotVisualizer.axes_pair_to_quaternion([0,0,1.], normal)
    settransform!(vis, :floor, p, Makie.Quaternion(q.v1, q.v2, q.v3, q.s))
    return nothing
end

function depthpixel_to_camera(p1, p2, depth, resolution, fovy)
	r1, r2 = resolution
	# pixel coordinate rescaled from -1 to 1
	αx = + (p1 * 2 / r1 - 1 / r1 - 1.0)
	αy = + (p2 * 2 / r2 - 1 / r2 - 1.0)

	# coordinate of the pixel in the camera frame
	# the pixel belongs to a plane 'depth plane' located at distance = depth from the camera.
	# l = half-size of the image projected on the depth plane.
	fovx = fovy * r1/r2
	lx = depth * tan(2π*fovx/360 / 2)
	ly = depth * tan(2π*fovy/360 / 2)

	# x, y positions in the depth plane
	x = αx * lx
	y = αy * ly
	# The z axis is directed towards the back of the camera when y points upwards and x points to the right
	return [x, y, -depth]
end


function camera_to_world(pc, eyeposition, lookat, up)
	# z axis = look_direction
	z = normalize(eyeposition - lookat)
	# y axis = up
	y = up - (up'*z)*z
	y = normalize(y)
	# x axis = y × z
	x = normalize(cross(y, z))
	# rotation matrix
	wRc = [x y z]
	pw = wRc * pc + eyeposition
	return pw
end

function depthpixel_to_world(px, py, depth, resolution, fovy, eyeposition, lookat, up)
	pc = depthpixel_to_camera(px, py, depth, resolution, fovy)
	pw = camera_to_world(pc, eyeposition, lookat, up)
	return pw
end





resolution0 = (600, 600)
vis = GLVisualizer1220(resolution=resolution0)
open(vis, visible=true)

set_floor!(vis)
object1 = HyperRectangle(Vec(0,0,0), Vec(0.2, 0.9, 1))
object2 = HyperRectangle(Vec(0,0,0), Vec(0.6, 0.2, 2.0))

setobject!(vis, :root, :object1, object1, color=RGBA(0,0,1,0.4))
setobject!(vis, :object1, :object2, object2, color=RGBA(1,0,0,0.4))

eyeposition0 = [0,2,5.0]
lookat0 = [1,00,00]
up0 = [0,0,1.0]
set_camera!(vis;
		eyeposition=eyeposition0,
		lookat=lookat0,
		up=up0,
		far=100.0,
		near=0.1,
		zoom=1.0,
		)

# settransform!(vis, :object1, [0,0,1.0], Quaternion(0,0,0,1.0))
# settransform!(vis, :object1, [0,0,1.0], Quaternion(sqrt(2)/2,0,0,sqrt(2)/2))
# settransform!(vis, :object1, [0,0,1.0], Quaternion(1,0,0,0.0))

depth_color = depth_buffer(vis)
maximum(depth_color)
minimum(depth_color)

depth_color = depth_buffer!(depth_color, vis)
maximum(depth_color)
minimum(depth_color)

px0 = 160
py0 = 160
depth0 = rotr90(depth_color)[py0, px0]
fovy0 = 45.0

pc0 = depthpixel_to_camera(px0, py0, depth0, resolution0, fovy0)
pw0 = depthpixel_to_world(px0, py0, depth0, resolution0, fovy0, eyeposition0, lookat0, up0)

pixel1 = HyperSphere(Point{3}(0,0,0.0), 0.007)
setobject!(vis, :root, Symbol(:pixel, counter), pixel1, color=RGBA(0.0,0.0,0.0,1.0))
settransform!(vis, Symbol(:pixel,counter), pw0, Makie.Quaternion(0,0,0,1.0))
@show

# Plots.spy(10*rotl90(depth_color))
linear_depth_color = (depth_color .- minimum(depth_color)) ./ (maximum(depth_color) - minimum(depth_color))
point_depth_color = deepcopy(linear_depth_color)

counter = 0
for j = 0:14
	for i = 0:14
		counter += 1
		@show counter
		px0 = 100 + 32i
		py0 = 100 + 32j
		point_depth_color[px0 .+ (-3:3), py0 .+ (-3:3)] .= 1
		depth0 = depth_color[px0, py0]

		pw0 = depthpixel_to_world(px0, py0, depth0, resolution0, fovy0, eyeposition0, lookat0, up0)

		pixel1 = HyperSphere(Point{3}(0,0,0.0), 0.025)
		setobject!(vis, :root, Symbol(:pixel, counter), pixel1, color=RGBA(0.0,0.0,0.0,1.0))
		settransform!(vis, Symbol(:pixel,counter), pw0, Makie.Quaternion(0,0,0,1.0))
		@show pc0
		@show pw0
	end
end

Plots.plot(Gray.(1 .- linear_depth_color))
Plots.plot(Gray.(1 .- rotl90(point_depth_color)))
linear_depth_color
maximum(depth_color)
minimum(depth_color)
maximum(linear_depth_color)
minimum(linear_depth_color)
