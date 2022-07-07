# struct Solver{T,X,E,EX,EP,B,BX,P,PX,PXI,K}
struct Solver{T,X,B,BX,P,PX,PXI,K}
    problem::ProblemData{T,X}
    # methods::ProblemMethods{T,E,EX,EP}
    methods::AbstractProblemMethods{T}
    cone_methods::ConeMethods{B,BX,P,PX,PXI,K}
    data::SolverData{T}

    solution::Point{T}
    candidate::Point{T}
    parameters::Vector{T}

    indices::Indices
    dimensions::Dimensions

    # linear_solver::LDLSolver{T,Int}
    linear_solver::LUSolver{T}

    step_sizes::StepSize{T}
    central_paths::CentralPath{T}
    fraction_to_boundary::Vector{T}
    penalty::Vector{T}
    dual::Vector{T}

    primal_regularization::Vector{T}
    primal_regularization_last::Vector{T}
    dual_regularization::Vector{T}

    options::Options{T}
    trace::Trace{T}
end

function Solver(equality, num_primals::Int, num_cone::Int;
    parameters=zeros(0),
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    custom=nothing,
    methods=nothing,
    options=Options(),
    )

    num_parameters = length(parameters)

    # dimensions
    dim = Dimensions(num_primals, num_cone, num_parameters;
        nonnegative=length(nonnegative_indices),
        second_order=[length(idx_soc) for idx_soc in second_order_indices])

    # indices
    idx = Indices(num_primals, num_cone, num_parameters;
        nonnegative=nonnegative_indices,
        second_order=second_order_indices)

    # codegen methods
    (methods == nothing) && (methods = ProblemMethods(equality, dim, idx))

    # cone methods
    cone_methods = ConeMethods(num_cone, nonnegative_indices, second_order_indices)

    # problem data
    p_data = ProblemData(dim.variables, num_parameters, dim.equality, num_cone;
        custom=custom)

    # solver data
    s_data = SolverData(dim, idx)

    # points
    solution = Point(dim, idx)
    candidate = Point(dim, idx)

    # interior-point
    step_sizes = StepSize(num_cone)
    central_paths = CentralPath(nonnegative_indices,
        second_order_indices, options.complementarity_tolerance)
    fraction_to_boundary = max.(0.99, 1.0 .- central_paths.central_path)

    # augmented Lagrangian
    penalty = [10.0]
    dual = zeros(dim.equality)

    # linear solver TODO: constructor
    random_solution = rand(dim.variables)
    # random_solution.all .= randn(dim.total)

    # evaluate!(p_data, methods, idx, random_solution, parameters,
    #     objective_jacobian_variables_variables=true,
    #     equality_jacobian_variables=true,
    #     equality_dual_jacobian_variables_variables=true,
    #     cone_jacobian_variables=true,
    #     cone_dual_jacobian_variables_variables=true,
    # )
    # cone!(p_data, cone_methods, idx, random_solution,
    #     jacobian=true,
    # )
    # residual_jacobian_variables!(s_data, p_data, idx, rand(1), rand(1), randn(num_equality), 1.0e-5, 1.0e-5,
    #     constraint_tensor=options.constraint_tensor)
    # residual_jacobian_variables_symmetric!(s_data.jacobian_variables_symmetric, s_data.jacobian_variables, idx,
    #     p_data.second_order_jacobians, p_data.second_order_jacobians)
    #
    linear_solver = options.compressed_search_direction ?
        lu_solver(s_data.dense_compressed_jacobian_variables) :
        lu_solver(s_data.dense_jacobian_variables)

    # regularization
    primal_regularization = [0.0]
    primal_regularization_last = [0.0]
    dual_regularization = [0.0]

    trace = Trace()

    Solver(
        p_data,
        methods,
        cone_methods,
        s_data,
        solution,
        candidate,
        parameters,
        idx,
        dim,
        linear_solver,
        step_sizes,
        central_paths,
        fraction_to_boundary,
        penalty,
        dual,
        primal_regularization,
        primal_regularization_last,
        dual_regularization,
        options,
        trace,
    )
end
