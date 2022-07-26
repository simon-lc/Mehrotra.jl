function get_polytope_bundle(;
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

    Af = [0.0  +1.0]
    bf = [0.0]
    Ap1 = [
        1.0  0.0;
        0.0  1.0;
        -1.0  0.0;
        0.0 -1.0;
        ] .- 0.30ones(4,2);
    bp1 = 0.2*[
        +1,
        +1,
        +1,
        1,
        ];
    Ap2 = [
        1.0  0.0;
        0.0  1.0;
        -1.0  0.0;
        0.0 -1.0;
        ] .+ 0.20ones(4,2);
    bp2 = 0.2*[
        -0.5,
        +1,
        +1.5,
        1,
        ];
    Ac = [
         1.0  0.0;
         0.0  1.0;
        -1.0  0.0;
         0.0 -1.0;
        ] .+ 0.20ones(4,2);
    bc = 0.2*[
        1,
        1,
        1,
        1,
        ];

    # nodes
    bodies = [
        Body183(timestep, mass, inertia, [Ap1, Ap2], [bp1, bp2], gravity=+gravity, name=:pbody),
        Body183(timestep, mass, inertia, [Ac], [bc], gravity=+gravity, name=:cbody),
        ]
    contacts = [
        PolyPoly183(bodies[1], bodies[2],
            friction_coefficient=friction_coefficient,
            name=:contact_1),
        PolyPoly183(bodies[1], bodies[2],
            parent_collider_id=2,
            friction_coefficient=friction_coefficient,
            name=:contact_2),
        PolyHalfSpace183(bodies[1], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p1),
        PolyHalfSpace183(bodies[1], Af, bf,
            parent_collider_id=2,
            friction_coefficient=friction_coefficient,
            name=:halfspace_p2),
        PolyHalfSpace183(bodies[2], Af, bf,
            friction_coefficient=friction_coefficient,
            name=:halfspace_c),
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
