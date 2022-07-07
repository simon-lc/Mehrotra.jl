function evaluate!(problem::ProblemData{T},
        methods::ProblemMethods{T,E,EX,EP},
        cone_methods::ConeMethods{B,BX,P,PX,PXI,TA},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        cone_constraint=false,
        cone_jacobian=false,
        cone_jacobian_inverse=false,
        ) where {T,E,EX,EP,B,BX,P,PX,PXI,TA}

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
        for (i, idx) in enumerate(methods.equality_jacobian_variables_sparsity)
            problem.equality_jacobian_variables[idx...] = methods.equality_jacobian_variables_cache[i]
        end
    end

    if (equality_jacobian_parameters && ne > 0 && nθ > 0)
        methods.equality_jacobian_parameters(methods.equality_jacobian_parameters_cache, x, θ)
        for (i, idx) in enumerate(methods.equality_jacobian_parameters_sparsity)
            problem.equality_jacobian_parameters[idx...] = methods.equality_jacobian_parameters_cache[i]
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
