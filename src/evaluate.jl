function evaluate!(problem::ProblemData{T},
        methods::AbstractProblemMethods{T,E,EC,EX,EXC,EP,EK,C,CC,S},
        cone_methods::ConeMethods{T,B,BX,P,PX},
        solution::Point{T},
        parameters::Vector{T};
        equality_constraint=false,
        equality_jacobian_variables=false,
        equality_jacobian_parameters=false,
        equality_jacobian_keywords=nothing,
        cone_constraint=false,
        cone_jacobian=false,
        sparse_solver::Bool=false,
        compressed::Bool=false,
        ) where {T,E,EC,EX,EXC,EP,EK,C,CC,S,B,BX,P,PX}

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
        # for each keyword
        if equality_jacobian_keywords != nothing
            for (i,k) in enumerate(equality_jacobian_keywords)
                func = methods.equality_jacobian_keywords[i]
                indices = methods.equality_jacobian_keywords_indices[i]
                cache = methods.equality_jacobian_parameters_cache[indices]
                func(cache, x, θ)
                problem.equality_jacobian_parameters.nzval[indices] .=
                    methods.equality_jacobian_parameters_cache[indices]
            end
        end

        methods.equality_jacobian_parameters(
            methods.equality_jacobian_parameters_cache, x, θ)
        problem.equality_jacobian_parameters.nzval .=
            methods.equality_jacobian_parameters_cache
    end

    # evaluate candidate cone product constraint, cone target and jacobian
    cone!(problem, cone_methods, solution,
        cone_constraint=cone_constraint,
        cone_jacobian=cone_jacobian,
        sparse_solver=sparse_solver,
    )
    return
end
