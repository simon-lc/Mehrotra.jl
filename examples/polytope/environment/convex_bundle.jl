using Polyhedra
using MeshCat
using RobotVisualizer
using StaticArrays
using Quaternions

include("../src/polytope.jl")
include("../src/rotate.jl")
include("../src/quaternion.jl")
include("../src/node.jl")
include("../src/body.jl")
include("../src/poly_poly.jl")
include("../src/poly_halfspace.jl")
include("../src/mechanism.jl")
include("../src/simulate.jl")
include("../src/visuals.jl")

################################################################################
# demo
################################################################################
# parameters

function get_convex_bundle(; 
    timestep=0.05, 
    gravity=-9.81, 
    mass=1.0, 
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.9,
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
        Body182(timestep, mass, inertia, [Ap1, Ap2], [bp1, bp2], gravity=+gravity, name=:pbody),
        Body182(timestep, mass, inertia, [Ac], [bc], gravity=+gravity, name=:cbody),
        ]
    contacts = [
        PolyPoly182(bodies[1], bodies[2], 
            friction_coefficient=friction_coefficient, 
            name=:contact_1),
        PolyPoly182(bodies[1], bodies[2], 
            parent_collider_id=2, 
            friction_coefficient=friction_coefficient, 
            name=:contact_2),
        PolyHalfSpace182(bodies[1], Af, bf, 
            friction_coefficient=friction_coefficient, 
            name=:halfspace_p1),
        PolyHalfSpace182(bodies[1], Af, bf, 
            parent_collider_id=2, 
            friction_coefficient=friction_coefficient, 
            name=:halfspace_p2),
        PolyHalfSpace182(bodies[2], Af, bf, 
            friction_coefficient=friction_coefficient, 
            name=:halfspace_c),
        ]
    indexing!([bodies; contacts])

    local_mechanism_residual(primals, duals, slacks, parameters) = 
        mechanism_residual(primals, duals, slacks, parameters, bodies, contacts)

    mechanism = Mechanism182(local_mechanism_residual, bodies, contacts, options=options)

    initialize_solver!(mechanism.solver)
    return mechanism
end
