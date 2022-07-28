
# x = scatter(1:4)
# screen = display(x)
# depth_color = GLMakie.depthbuffer(screen)
# # Look at result:
# heatmap(depth_color, colormap=:grays)

# using Pkg
# Pkg.add("GLMakie")
# Pkg.add("FileIO")
# Pkg.add("Downloads")
# Pkg.add("JSON")
# Pkg.add("Makie")
# Pkg.add("GeometryBasics")
# Pkg.add("Colors")
# Pkg.add("LinearAlgebra")

using GLMakie
using FileIO
using Downloads
using JSON
using Makie.GeometryBasics: Pyramid
using GeometryBasics
using Colors
using LinearAlgebra
using Plots

pyr = Pyramid(GeometryBasics.Point3f(0), 1.0f0, 1.0f0)
rectmesh = Rect3(GeometryBasics.Point3f(-0.5), GeometryBasics.Vec3f(1))
sphere = Sphere(GeometryBasics.Point3f(-0.5), 1)
Cone(; quality=10) = merge([
    Makie._circle(GeometryBasics.Point3f(0), 0.5f0, GeometryBasics.Vec3f(0, 0, -1), quality),
    Makie._mantle(GeometryBasics.Point3f(0), GeometryBasics.Point3f(0, 0, 1), 0.5f0, 0.0f0, quality)])
cone = Cone()

brain = load(assetpath("brain.stl"))
matball = load(assetpath("matball_base.obj"))
matball_inner = load(assetpath("matball_inner.obj"))
matball_outer = load(assetpath("matball_outer.obj"))
matcapIds = Downloads.download("https://github.com/lazarusA/BeautifulMakie/tree/main/_assets/data/matcapIds.json", "/home/simon/Downloads/glmakie.json")
# download the data from here:
# https://github.com/lazarusA/BeautifulMakie/tree/main/_assets/data
ids = JSON.parsefile("/home/simon/Downloads/glmakie.json")

function plotmat()
    idx = Observable(1)
    idpng = @lift(ids[$idx])
    matcap = @lift(load(Downloads.download("https://raw.githubusercontent.com/nidorx/matcaps/master/1024/$($idpng).png")))
    ambient = GeometryBasics.Vec3f(0.8, 0.8, 0.8)
    shading = true
    fig = Figure(resolution=(1200, 900))
    axs = [LScene(fig[i, j]; show_axis=false)
           for j in 1:3, i in 1:2]
    mesh!(axs[5], sphere; matcap, shading, ambient)
    mesh!(axs[3], rectmesh; matcap, shading, ambient, transparency=true)
    mesh!(axs[4], pyr; matcap, shading, ambient)
    mesh!(axs[2], matball; matcap, shading, ambient)
    mesh!(axs[2], matball_inner; matcap, shading, ambient)
    mesh!(axs[2], matball_outer; matcap, shading, ambient)
    mesh!(axs[6], cone; matcap, shading, ambient)
    mesh!(axs[1], brain; matcap, shading, ambient)
    GLMakie.rotate!(axs[2].scene, 2.35)
    center!(axs[2].scene)
    zoom!(axs[2].scene, cameracontrols(axs[2].scene), 0.75)
    zoom!(axs[3].scene, cameracontrols(axs[3].scene), 1.2)
    zoom!(axs[4].scene, cameracontrols(axs[4].scene), 1.2)
    sl = Slider(fig[1:2, 4], range=1:length(ids), startvalue=582, horizontal=false)
    connect!(idx, sl.value)
    fig[0, 1:3] = Label(fig, @lift("idnum [$(1*$idx)] png material : $(ids[$idx])"), textsize=20,
        tellheight=true, tellwidth=false)
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
