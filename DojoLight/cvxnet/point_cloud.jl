function point_cloud(vis::GLVisualizer.Visualizer, mechanism::Mechanism1170, z,
        p1, p2, resolution, eyeposition, lookat, up)

    GLVisualizer.set_camera!(vis;
        eyeposition=eyeposition,
        lookat=lookat,
        up=up)

    set_mechanism!(vis, mechanism, z)

    n1 = length(p1)
    n2 = length(p2)

    depth = zeros(Float32, resolution...)
    world_coordinates = zeros(3, n1 * n2)

    depth_buffer!(depth, glvis)
    depthpixel_to_world!(world_coordinates, depth, p1, p2, vis)

    return world_coordinates
end
