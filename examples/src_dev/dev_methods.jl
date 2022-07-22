using Random
using StaticArrays
using LoopVectorization

include("../examples/benchmark_problems/lcp_utils.jl")

################################################################################
# coupled constraints
################################################################################
# dimensions
num_primals = 10
num_cone = 10
num_parameters = num_primals^2 + num_primals + num_cone^2 + num_cone

# cone type
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

# Jacobian
Random.seed!(0)
As = rand(num_primals, num_primals)
A = As' * As
B = rand(num_primals, num_cone)
C = B'
d = rand(num_primals)
e = zeros(num_cone)
parameters = [vec(A); vec(B); vec(C); d; e]

num_parameters = length(parameters)
indices = Indices(num_primals, num_cone, num_parameters)
dimensions = Dimensions(num_primals, num_cone, num_parameters)
# meths = structured_symbolics_methods(lcp_residual, dimensions, indices)


# solver
solver = Solver(lcp_residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(sparse_solver=true, max_iterations=10)
    )

solver.problem
solver.data
# solve
Mehrotra.solve!(solver)
solver.data.jacobian_variables_sparse.matrix[end-3:end,end-3:end]

solver.problem

function evaluate!(problem::ProblemData{T},
        methods::StructuredProblemMethods112{T,O,OY,OZ,OP,S,SY,SS,SP},
        cone_methods::ConeMethods{T,B,BX,P,PX,PXI},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        cone_jacobian_inverse=false,
        ) where {T,O,OY,OZ,OP,S,SY,SS,SP,B,BX,P,PX,PXI}

    x = solution.all
    y = solution.primals
    z = solution.duals
    s = solution.slacks
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    no = length(problem.optimality_constraint)
    ns = length(problem.slackness_constraint)

    (equality_constraint && no > 0) && methods.optimality_constraint(problem.optimality_constraint, x, θ)
    (equality_constraint && ns > 0) && methods.slackness_constraint(problem.slackness_constraint, x, θ)

    if (optimality_jacobian_variables && ne > 0)
        methods.optimality_jacobian_variables(methods.optimality_jacobian_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.optimality_jacobian_variables_sparsity)
            problem.optimality_jacobian_variables[idx...] = methods.optimality_jacobian_variables_cache[i]
        end
    end
    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] = methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && no > 0 && nθ > 0)
        methods.optimality_jacobian_parameters(methods.optimality_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.optimality_jacobian_parameters_sparsity)
            problem.optimality_jacobian_parameters[idx...] = methods.optimality_jacobian_parameters_cache[i]
        end
    end
    if (equality_jacobian_parameters && ns > 0 && nθ > 0)
        methods.slackness_jacobian_parameters(methods.slackness_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.slackness_jacobian_parameters_sparsity)
            problem.slackness_jacobian_parameters[idx...] = methods.slackness_jacobian_parameters_cache[i]
        end
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
        cone_jacobian_inverse=cone_jacobian_inverse,
    )
    return
end
# n = 35
#
# As = sprand(n, n, 0.10)
# As = As + As'
# lu(As)
# Ad = Matrix(As)
# A_sparsity = collect(zip([findnz(As)[1:2]...]...))
# A_cartesian = [CartesianIndex(a...) for a in A_sparsity]
# A_integer = [a[1] + (a[2]-1)*n for a in A_sparsity]
# Av = zeros(length(A_sparsity))
# Av .= As.nzval
# Av
#
# Bs = similar(As)
# Bd = zeros(n,n)
#
# function fillin0!(Av::Vector{T}, A_sparsity::Vector{Tuple{Int,Int}}, Bd::Matrix{T}) where T
#     for (i, idx) in enumerate(A_sparsity)
#         Bd[idx...] = Av[i]
#     end
#     return nothing
# end
# function fillin1!(Av::Vector{T}, Bs::SparseMatrixCSC{T,Int}) where T
#     Bs.nzval .= Av
#     return nothing
# end
# function fillin2!(Av::Vector{T}, A_sparsity::Vector{CartesianIndex{2}}, Bd::Matrix{T}) where T
#     for (i, idx) in enumerate(A_sparsity)
#         Bd[idx] = Av[i]
#     end
#     return nothing
# end
# function fillin3!(Av::Vector{T}, A_sparsity::Vector{Int}, Bd::Matrix{T}) where T
#     for (i, idx) in enumerate(A_sparsity)
#         Bd[idx] = Av[i]
#     end
#     return nothing
# end
#
#
#
#
#
# fillin0!(Av, A_sparsity, Bd)
# Main.@code_warntype fillin0!(Av, A_sparsity, Bd)
# @benchmark $fillin0!($Av, $A_sparsity, $Bd)
#
# fillin1!(Av, Bs)
# Main.@code_warntype fillin1!(Av, Bs)
# @benchmark $fillin1!($Av, $Bs)
#
# fillin2!(Av, A_cartesian, Bd)
# Main.@code_warntype fillin2!(Av, A_cartesian, Bd)
# @benchmark $fillin2!($Av, $A_cartesian, $Bd)
#
# fillin3!(Av, A_integer, Bd)
# Main.@code_warntype fillin3!(Av, A_integer, Bd)
# @benchmark $fillin3!($Av, $A_integer, $Bd)

# lu_factorization = ilu0(As)
# ilu0!(lu_factorization, As)
# @benchmark $ilu0!($lu_factorization, $As)
#
# lu(As)
# @benchmark $lu($As)
#
# lu(Ad)
# @benchmark $lu($Ad)
