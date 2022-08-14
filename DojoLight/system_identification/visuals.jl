function visualize!(vis::Visualizer, context::CvxContext1290, measurements::Vector{<:Measurement};
        animation::MeshCat.Animation=MeshCat.Animation(), name::Symbol=:context)

    mechanism = context.mechanism
    cameras = context.cameras

    # build point cloud
    num_points = size.(measurements[1].p, 2)
    build_point_cloud!(vis[name], num_points)
    eye_positions = [c.eye_position for c in cameras]

    # animate point cloud
    for (i, measurement) in enumerate(measurements)
        MeshCat.atframe(animation, i) do
            set_2d_point_cloud!(vis[name], eye_positions, measurement.p)
        end
    end
    # set animation
    MeshCat.setanimation!(vis, animation)

    # add robot visualization
    _, animation = visualize!(vis[name], mechanism, [m.z for m in measurements], animation=animation)
    return vis, animation
end
