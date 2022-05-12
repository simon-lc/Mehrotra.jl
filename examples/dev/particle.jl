using Mehrotra

function unpack_parameters(parameters)
    p2 = parameters[1:3]
    v15 = parameters[4:6]
    u = parameters[7:9]
    timestep = parameters[10]
    mass = parameters[11]
    gravity = parameters[12]
    friction_coefficient = parameters[13]
    side = parameters[14]
    return p2, v15, u, timestep, mass, gravity, friction_coefficient, side
end

function residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_parameters(parameters)

    v25 = y
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    vtan = v25[2:3]

    γ = z[1:1]
    ψ = z[2:2]
    β = z[3:6]

    sγ = s[1:1]
    sψ = s[2:2]
    sβ = s[3:6]

    N = [0 0 1]
    D = [1 0 0;
         0 1 0]
    P = [+D;
         -D]

    res = [
        mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - P' * β - u * timestep;
        sγ - (p3[3:3] .- side/2);
        sψ - (friction_coefficient * γ - [sum(β)]);
        sβ - (P * v25 + ψ[1]*ones(4));
        # Γ .* S .- κ[1];
        ]
    return res
end

num_primals = 3
num_cone = 6
num_parameters = 14
idx_nn = collect(1:6)
idx_soc = [collect(1:0)]
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = rand(3)
parameters = [p2; v15; u; 0.01; 1.0; -9.81; 0.3; 0.5]


dim = Dimensions(num_primals, num_cone, num_parameters;
    nonnegative=num_cone,
    second_order=[0,])
idx = Indices(num_primals, num_cone, num_parameters;
    nonnegative=collect(1:num_cone),
    second_order=[collect(1:0)])
meths = generate_gradients(residual, dim, idx)
prob_methods = ProblemMethods(residual, dim, idx)
prob_data = ProblemData(dim.variables, dim.parameters, dim.equality, dim.duals)
solver_data = SolverData(dim, idx)
solver = Solver(residual, num_primals, num_cone, parameters=parameters)

solver_data
prob_data


solver = Solver(residual, num_primals, num_cone, parameters=parameters)
solver.options.residual_tolerance = 1e-6
solver.options.complementarity_tolerance = 1e-4
solver.options.max_iterations = 30
solver.options.max_iteration_line_search = 10
solver.options.verbose = true
solve!(solver)
solver.solution
solver.candidate

solver.data.residual

solver.solution
JV = solver.data.jacobian_variables
plot(Gray.(100abs.(Matrix(JV))))

JV[10:15,4:9]
JV[10:15,10:15]
JV[4:9,10:15]


function residual_complex(solver, variables)
    solver.solution.all .= variables
    # evaluate
    evaluate!(solver.problem, solver.methods, solver.solution, solver.parameters,
        equality_constraint=true,
        equality_jacobian_variables=true,
    )

    cone!(solver.problem, solver.cone_methods, solver.solution,
        product=true,
        jacobian=true,
        target=true
    )

    # residual
    residual!(solver.data, solver.problem, solver.indices, solver.solution, [solver.options.complementarity_tolerance])

    # violations
    equality_violation = norm(solver.data.residual.equality, Inf)
    cone_product_violation = norm(solver.data.residual.cone_product, Inf)
    return deepcopy(solver.data.residual.all), solver.data.jacobian_variables
end

v0 = rand(dim.variables)
residual_complex(solver, v0)
JV0 = FiniteDiff.finite_difference_jacobian(variables -> residual_complex(solver, variables)[1], v0)
JV1 = residual_complex(solver, v0)[2]
norm(JV0[1:3,1:15] - JV1[1:3,1:15])
norm(JV0[4:9,1:15] - JV1[4:9,1:15])
norm(JV0[10:15,1:15] - JV1[10:15,1:15])
norm(JV0 - JV1)
JV0[10:15,1:3]
JV1[10:15,1:3]

JV0[10:15,4:9]
JV1[10:15,4:9]

JV0[10:15,10:15]
JV1[10:15,10:15]
solver.problem.cone_product_jacobian_dual
solver.problem.cone_product_jacobian_slack

solver.candidate
solver.data


cone!(solver.problem, solver.cone_methods, solver.solution,
    product=true,
    jacobian=true,
    target=true
)
solver.cone_methods.product_jacobian(solver.problem.cone_product_jacobian_dual, ones(6), ones(6))
solver.solution
solver.problem.cone_product_jacobian_dual
solver
