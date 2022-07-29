"""
    set_floor!(vis; x, y, z, origin, normal, color, tilepermeter, imagename, axis, grid)
    adds floor to visualization
    vis::Visualizer
    x: lateral position
    y: longitudinal position
    z: vertical position
	origin:: position of the center of the floor
    normal:: unit vector indicating the normal to the floor
    color: RGBA
    tilepermeter: scaling
    imagename: path to image
    axis: flag to turn on visualizer axis
    grid: flag to turn on visualizer grid
"""
function set_floor!(vis::GLVisualizer1160;
	    x=20.0,
	    y=20.0,
	    z=0.1,
	    origin=[0,0,0.0],
		normal=[0,0,1.0],
	    color=RGBA(0.5,0.5,0.5,1.0),
	    axis::Bool=false,
	    grid::Bool=false)
	obj = HyperRectangle(Vec(-x/2, -y/2, -z), Vec(x, y, z))
	setobject!(vis, :root, obj, color=color, name=:floor)

	p = origin
	q = RobotVisualizer.axes_pair_to_quaternion([0,0,1.], normal)
    settransform!(vis, :floor, p, Quaternion(q.v1, q.v2, q.v3, q.s))
    return nothing
end

vis = GLVisualizer1160()
open(vis)

set_floor!(vis)

sphere_plot = mesh!(scene, Sphere(Point3f(0, 0, 0), 0.1), color=:black)
sphere_plot = mesh!(scene, Sphere(Point3f(1, 0, 0), 0.1), color=:red)
sphere_plot = mesh!(scene, Sphere(Point3f(0, 1, 0), 0.1), color=:green)
sphere_plot = mesh!(scene, Sphere(Point3f(0, 0, 1), 0.1), color=:blue)
x = 10
y = 10
z = 0.1
obj = HyperRectangle(Vec(-x/2, -y/2, -z), Vec(x, y, z))
sphere_plot = mesh!(scene, obj, color=RGBA(0.4, 0.4, 0.4, 1.0))
