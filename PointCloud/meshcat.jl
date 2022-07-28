using GLMakie
using FileIO
using Downloads
using JSON
using Makie.GeometryBasics: Pyramid
using GeometryBasics
using Colors
using LinearAlgebra
using Plots

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
