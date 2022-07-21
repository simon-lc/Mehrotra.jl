function evaluate!(problem::ProblemData{T},
        methods::AbstractProblemMethods{T,E,EC,EX,EXC,EP,C,CC,S},
        cone_methods::ConeMethods{T,B,BX,P,PX,PXI},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        cone_jacobian_inverse=false,
        sparse_solver::Bool=false,
        compressed::Bool=false,
        ) where {T,E,EC,EX,EXC,EP,C,CC,S,B,BX,P,PX,PXI}

    x = solution.all
    θ = parameters

    # dimensions
    nθ = length(θ)

    # equality
    ne = length(problem.equality_constraint)

    if equality_constraint && ne > 0
        if compressed
            methods.equality_constraint_compressed(
                problem.equality_constraint_compressed, x, θ)
        end
        # always useful for violation evaluation
        methods.equality_constraint(
            problem.equality_constraint, x, θ)
    end

    if (equality_jacobian_variables && ne > 0)
        if compressed
            methods.equality_jacobian_variables_compressed(
                methods.equality_jacobian_variables_compressed_cache, x, θ)
            problem.equality_jacobian_variables_compressed.nzval .=
                methods.equality_jacobian_variables_compressed_cache
        else
            methods.equality_jacobian_variables(
                methods.equality_jacobian_variables_cache, x, θ)
            problem.equality_jacobian_variables.nzval .=
                methods.equality_jacobian_variables_cache
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(
            methods.equality_jacobian_parameters_cache, x, θ)
        problem.equality_jacobian_parameters.nzval .=
            methods.equality_jacobian_parameters_cache
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
        cone_jacobian_inverse=cone_jacobian_inverse,
        sparse_solver=sparse_solver,
    )
    return
end



# function evaluate!(problem::ProblemData112{T},
#         methods::AbstractProblemMethods{T,E,EX,EP},
#         cone_methods::ConeMethods{B,BX,P,PX,PXI},
#         solution::Point{T},
#         parameters::Vector{T};
#         equality_constraint=false,
#         equality_jacobian_variables=false,
#         equality_jacobian_parameters=false,
#         cone_constraint=false,
#         cone_jacobian=false,
#         cone_jacobian_inverse=false,
#         ) where {T,E,EX,EP,B,BX,P,PX,PXI}
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
#     )
#     return
# end



# using Random
# solver = random_lcp(options=Options(
#     verbose=false,
#     ))
#
# solver.solution.all .= rand(solver.dimensions.variables)
# problem = solver.problem
# meths = solver.methods
# cone_methods = solver.cone_methods
# solution = solver.solution
# parameters = solver.parameters
# evaluate!(problem, meths, cone_methods, solution, parameters,
#     equality_constraint=true,
#     equality_jacobian_variables=true,
#     equality_jacobian_parameters=true,
#     cone_constraint=true,
#     cone_jacobian=true,
#     cone_jacobian_inverse=true,
#     )
#
# problem
# @benchmark $evaluate!($problem, $meths, $cone_methods, $solution, $parameters,
#     equality_constraint=true,
#     equality_jacobian_variables=true,
#     equality_jacobian_parameters=true,
#     cone_constraint=true,
#     cone_jacobian=true,
#     cone_jacobian_inverse=true,
#     sparse_solver=true
#     )
