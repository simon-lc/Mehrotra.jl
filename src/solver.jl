struct Solver{T,E,EC,EX,EXC,EP,EK,C,CC,S,B,BX,P,PX,X,L}
    problem::ProblemData{T,X}
    methods::ProblemMethods{T,E,EC,EX,EXC,EP,EK,C,CC,S}
    cone_methods::ConeMethods{T,B,BX,P,PX}
    data::SolverData{T}

    solution::Point{T}
    candidate::Point{T}
    parameters::Vector{T}

    indices::Indices
    dimensions::Dimensions

    linear_solver::L

    step_sizes::StepSize{T}
    central_paths::CentralPath{T}
    fraction_to_boundary::Vector{T}
    penalty::Vector{T}
    dual::Vector{T}

    primal_regularization::Vector{T}
    primal_regularization_last::Vector{T}
    dual_regularization::Vector{T}

    consistency::Consistency
    options::Options{T}
    trace::Trace{T}
end

function Solver(equality, num_primals::Int, num_cone::Int;
    parameters=zeros(0),
    nonnegative_indices=collect(1:num_cone),
    second_order_indices=[collect(1:0)],
    parameter_keywords=Dict{Symbol,Vector{Int}}(:all => 1:length(parameters)),
    custom=nothing,
    methods=nothing,
    method_type::Symbol=:symbolic, #:finite_difference
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
        second_order=second_order_indices,
        parameter_keywords=parameter_keywords)

    # codegen methods
    if methods == nothing
        if method_type == :symbolic
            methods = symbolics_methods(equality, dim, idx,
                primal_regularization=options.primal_regularization,
                dual_regularization=options.dual_regularization)
        elseif method_type == :finite_difference
            methods = finite_difference_methods(equality, dim, idx,
                primal_regularization=options.primal_regularization,
                dual_regularization=options.dual_regularization)
        else
            error("unknown method_type")
        end
    end

    # cone methods
    cone_methods = ConeMethods(num_cone, nonnegative_indices, second_order_indices)

    # problem data
    problem = ProblemData(dim.variables, num_parameters, dim.equality, num_cone;
        custom=custom)
    allocate_sparse_matrices!(problem, methods, cone_methods)

    # solver data
    data = SolverData(dim, idx, problem)
    allocate_sparse_matrices!(data, methods)

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
    random_solution = Point(dim, idx)
    random_solution.all .= rand(dim.variables)

    residual!(data, problem, idx, #central_paths.central_path;
        compressed=options.compressed_search_direction,
        sparse_solver=options.sparse_solver)

    if options.sparse_solver
        if options.compressed_search_direction
            linear_solver = options.symmetric ?
                ldl_solver(data.jacobian_variables_compressed_sparse) :
                sparse_lu_solver(data.jacobian_variables_compressed_sparse + 1e3I)
        else
            linear_solver = sparse_lu_solver(data.jacobian_variables_sparse.matrix + 1e3I)
        end
    else
        linear_solver = options.compressed_search_direction ?
            lu_solver(data.jacobian_variables_compressed_dense) :
            lu_solver(data.jacobian_variables_dense)
    end

    # regularization
    primal_regularization = [0.0]
    primal_regularization_last = [0.0]
    dual_regularization = [0.0]

    consistency = Consistency()
    trace = Trace()

    Solver(
        problem,
        methods,
        cone_methods,
        data,
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
        consistency,
        options,
        trace,
    )
end


function allocate_sparse_matrices!(problem::ProblemData, methods::ProblemMethods,
        cone_methods::ConeMethods)

    for idx in methods.equality_jacobian_variables_sparsity
        problem.equality_jacobian_variables[idx...] = 1.0
    end
    for idx in methods.equality_jacobian_variables_compressed_sparsity
        problem.equality_jacobian_variables_compressed[idx...] = 1.0
    end
    for idx in methods.equality_jacobian_parameters_sparsity
        problem.equality_jacobian_parameters[idx...] = 1.0
    end


    for idx in cone_methods.product_jacobian_sparsity
        problem.cone_product_jacobian_duals[idx...] = 1.0
    end
    for idx in cone_methods.product_jacobian_sparsity
        problem.cone_product_jacobian_slacks[idx...] = 1.0
    end
    return nothing
end


function allocate_sparse_matrices!(data::SolverData, methods::ProblemMethods)

    for idx in methods.equality_jacobian_variables_sparsity
        data.jacobian_variables_sparse.matrix[idx...] = 1.0
    end
    for idx in methods.equality_jacobian_variables_compressed_sparsity
        data.jacobian_variables_compressed_sparse[idx...] = 1.0
    end
    for idx in methods.equality_jacobian_parameters_sparsity
        data.jacobian_parameters[idx...] = 1.0
    end
    return nothing
end
