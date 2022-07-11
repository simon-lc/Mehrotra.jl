function evaluate!(problem::ProblemData{T},
        methods::AbstractProblemMethods{T,E,EX,EP},
        cone_methods::ConeMethods{B,BX,P,PX,PXS,PXI,TA},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        cone_jacobian_inverse=false,
        sparse_solver::Bool=false,
        ) where {T,E,EX,EP,B,BX,P,PX,PXS,PXI,TA}

    x = solution.all
    y = solution.primals
    z = solution.duals
    s = solution.slacks
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    (equality_constraint && ne > 0) && methods.equality_constraint(problem.equality_constraint, x, θ)

    if (equality_jacobian_variables && ne > 0)
        methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
        if sparse_solver
            for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
                problem.equality_jacobian_variables[idx...] = methods.equality_jacobian_variables_cache[i]
            end
        else
            problem.equality_jacobian_variables_sparse.nzval .= methods.equality_jacobian_variables_cache
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
        if sparse_solver
            for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
                problem.equality_jacobian_parameters[idx...] = methods.equality_jacobian_parameters_cache[i]
            end
        else
            problem.equality_jacobian_parameters_sparse.nzval .= methods.equality_jacobian_parameters_cache
        end
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
        cone_jacobian_inverse=cone_jacobian_inverse,
        cone_target=true # TODO this should only be true once at the beginning of the solve
    )
    return
end

# function evaluate!(problem::ProblemData112{T},
#         methods::AbstractProblemMethods{T,E,EX,EP},
#         cone_methods::ConeMethods{B,BX,P,PX,PXS,PXI,TA},
#         solution::Point{T},
#         parameters::Vector{T};
#         equality_constraint=false,
#         equality_jacobian_variables=false,
#         equality_jacobian_parameters=false,
#         cone_constraint=false,
#         cone_jacobian=false,
#         cone_jacobian_inverse=false,
#         ) where {T,E,EX,EP,B,BX,P,PX,PXS,PXI,TA}
#
#     x = solution.all
#     y = solution.primals
#     z = solution.duals
#     s = solution.slacks
#     θ = parameters
#
#     # dimensions
#     nθ = length(θ)
#
#     # equality
#     ne = length(problem.equality_constraint)
#
#     (equality_constraint && ne > 0) && methods.equality_constraint(problem.equality_constraint, x, θ)
#
#     if (equality_jacobian_variables && ne > 0)
#         methods.equality_jacobian_variables(methods.equality_jacobian_variables_cache, x, θ)
#         problem.equality_jacobian_variables.nzval .= methods.equality_jacobian_variables_cache
#     end
#
#     if (equality_jacobian_parameters && ne > 0 && nθ > 0)
#         methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
#         problem.equality_jacobian_parameters.nzval .= methods.equality_jacobian_parameters_cache
#     end
#
#     # evaluate candidate cone product constraint, cone target and jacobian
#     cone!(problem, cone_methods, solution,
#         cone_constraint=cone_constraint,
#         cone_jacobian=cone_jacobian,
#         cone_jacobian_inverse=cone_jacobian_inverse,
#         cone_target=true # TODO this should only be true once at the beginning of the solve
#     )
#     return
# end



# using Random
# solver = random_lcp(options=Options(
#     verbose=false,
#     ))
#
#
# problem = solver.problem
# meths = solver.methods
# cone_methods = solver.cone_methods
# solution = solver.solution
# parameters = solver.parameters
# evaluate!(problem, meths, cone_methods, solution, parameters,
#     equality_constraint=false,
#     equality_jacobian_variables=false,
#     equality_jacobian_parameters=false,
#     cone_constraint=false,
#     cone_jacobian=false,
#     cone_jacobian_inverse=false,
#     )
#
# @benchmark $evaluate!($problem, $meths, $cone_methods, $solution, $parameters,
#     equality_constraint=true,
#     equality_jacobian_variables=true,
#     equality_jacobian_parameters=true,
#     cone_constraint=true,
#     cone_jacobian=true,
#     cone_jacobian_inverse=true,
#     )
#
