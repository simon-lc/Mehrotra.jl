function get_sphere_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
    method_type::Symbol=:finite_difference,
    options=Options(
        # verbose=false,
        complementarity_tolerance=1e-4,
        compressed_search_direction=false,
        max_iterations=30,
        sparse_solver=false,
        warm_start=true,
        )
    )

    sphere_radius = 0.2
    Af = [0.0  +1.0]
    bf = [0.0]

    # nodes
    bodies = [
        Body183(timestep, mass, inertia, [Af], [bf], gravity=+gravity, name=:pbody),
        ]
    contacts = [
        SphereHalfSpace1831(bodies[1], sphere_radius, Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p1),
        ]
    indexing!([bodies; contacts])

    local_mechanism_residual(primals, duals, slacks, parameters) =
        mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

    mechanism = Mechanism183(
        local_mechanism_residual,
        bodies,
        contacts,
        options=options,
        method_type=method_type)

    initialize_solver!(mechanism.solver)
    return mechanism
end
