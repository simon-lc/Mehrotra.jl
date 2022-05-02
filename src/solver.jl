struct Solver212{T,X,EX,EP,C,CX,CP,CD,CDX,CDXX,CDXP,B,BX,P,PX,PXI,K}
    problem::ProblemData212{T,X}
    methods::ProblemMethods{T,E,EX,EP,C,CX,CP,CD,CDX,CDXX,CDXP}
    cone_methods::ConeMethods{T,B,BX,P,PX,PXI,K}
    data::SolverData212{T}

    solution::Vector{T}
    candidate::Vector{T}
    parameters::Vector{T}

    indices::Indices212
    dimensions::Dimensions212

    linear_solver::LDLSolver{T,Int}

    central_path::Vector{T}
    fraction_to_boundary::Vector{T}
    penalty::Vector{T}
    dual::Vector{T}

    primal_regularization::Vector{T}
    primal_regularization_last::Vector{T}
    dual_regularization::Vector{T}

    options::Options{T}
end

function Solver(methods, num_variables, num_parameters, num_equality, num_cone;
    parameters=zeros(num_parameters),
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    custom=nothing,
    options=Options())

    # indices
    idx = Indices212(num_variables, num_parameters, num_cone;
        nonnegative=nonnegative_indices,
        second_order=second_order_indices)

    # dimensions
    dim = Dimensions(num_variables, num_parameters, num_equality, num_cone;
        nonnegative=length(nonnegative_indices),
        second_order=[length(idx_soc) for idx_soc in second_order_indices])

    # cone methods
    cone_methods = ConeMethods(num_cone, nonnegative_indices, second_order_indices)

    # problem data
    p_data = ProblemData(num_variables, num_parameters, num_equality, num_cone;
        nonnegative_indices=nonnegative_indices,
        second_order_indices=second_order_indices,
        custom=custom)

    # solver data
    s_data = SolverData(dim, idx,
        max_filter=options.max_filter)

    # points
    solution = Point(dim, idx)
    candidate = Point(dim, idx)

    # interior-point
    central_path = [0.1]
    fraction_to_boundary = [max(0.99, 1.0 - central_path[1])]

    # augmented Lagrangian
    penalty = [10.0]
    dual = zeros(num_equality)

    # linear solver TODO: constructor
    random_solution = Point(dim, idx)
    random_solution.all .= randn(dim.total)

    evaluate!(p_data, methods, idx, random_solution, parameters,
        objective_jacobian_variables_variables=true,
        equality_jacobian_variables=true,
        equality_dual_jacobian_variables_variables=true,
        cone_jacobian_variables=true,
        cone_dual_jacobian_variables_variables=true,
    )
    cone!(p_data, cone_methods, idx, random_solution,
        jacobian=true,
    )
    residual_jacobian_variables!(s_data, p_data, idx, rand(1), rand(1), randn(num_equality), 1.0e-5, 1.0e-5,
        constraint_tensor=options.constraint_tensor)
    residual_jacobian_variables_symmetric!(s_data.jacobian_variables_symmetric, s_data.jacobian_variables, idx,
        p_data.second_order_jacobians, p_data.second_order_jacobians)

    linear_solver = ldl_solver(s_data.jacobian_variables_symmetric)

    # regularization
    primal_regularization = [0.0]
    primal_regularization_last = [0.0]
    dual_regularization = [0.0]

    Solver212(
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
        central_path,
        fraction_to_boundary,
        penalty,
        dual,
        primal_regularization,
        primal_regularization_last,
        dual_regularization,
        options,
    )
end

function Solver(objective, equality, cone, num_variables::Int;
    parameters=zeros(0),
    nonnegative_indices=nothing,
    second_order_indices=nothing,
    custom=nothing,
    options=Options(),
    )

    # codegen methods
    num_parameters = length(parameters)
    methods, num_equality, num_cone = ProblemMethods(num_variables, num_parameters, objective, equality, cone)

    # solver
    solver = Solver(methods, num_variables, num_parameters, num_equality, num_cone;
        parameters=parameters,
        nonnegative_indices=(nonnegative_indices === nothing ? collect(1:num_cone) : nonnegative_indices),
        second_order_indices=(second_order_indices === nothing ? [collect(1:0)] : second_order_indices),
        custom=custom,
        options=options)

    return solver
end
