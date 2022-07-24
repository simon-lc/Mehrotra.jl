"""
    residual!(data, problem, idx, central_path; compressed, sparse_solver)
    compute non compressed residual and jacobian for problem structured as follows,
        jacobian_variables = [
            A B C
            D E F
            0 S Z
            ]
    compute compressed residual and jacobian for problem structured as follows,
        jacobian_variables = [
            A B 0
            C 0 D
            0 S Z
            ]
    where
        A = ∂optimality_constraints / ∂y == num_primals, num_primals
        B = ∂optimality_constraints / ∂z == num_primals, num_cone
        C = ∂slack_constraints / ∂y == num_cone, num_primals
        I = identity matrix == num_cone, num_cone
        S = ∂(z∘s) / ∂z == num_cone, num_cone
        Z = ∂(z∘s) / ∂s == num_cone, num_cone

    we solve
        |A B 0| |Δy|   |-optimality    |
        |C 0 D|×|Δz| = |-slack_equality|
        |0 S Z| |Δs|   |-cone_product  |
    we get the compressed form
        |A B      | |Δy|   |-optimality                        |
        |C -DZ⁻¹S |×|Δz| = |-slack_equality + DZ⁻¹ cone_product|
        Δs = -Z⁻¹ (cone_product + S * Δz)

    data: SolverData
    problem: ProblemData
    idx: Indices
    central_path: Vector{T}
    compressed: Bool

"""
function residual!(data::SolverData, problem::ProblemData, idx::Indices;
        residual=false,
        jacobian_variables=false,
        jacobian_parameters=false,
        compressed::Bool=false,
        sparse_solver::Bool=false)

    # residual
    if residual
        if compressed
            fill!(data.residual_compressed.all, 0.0) # reset
            # equality
            data.residual_compressed.equality .= problem.equality_constraint_compressed
        end
        # this is always useful for violation evaluation
        fill!(data.residual.all, 0.0) # reset
        # equality
        data.residual.equality .= problem.equality_constraint
        # cone: z ∘ s ####- κ e
        for (i, ii) in enumerate(idx.cone_product)
            data.residual.all[ii] = problem.cone_product[i]
        end
    end


    # jacobian variables
    if jacobian_variables
        # equality
        if !compressed && sparse_solver
            fill!(data.jacobian_variables_sparse, problem.equality_jacobian_variables, :equality_jacobian_variables)
        elseif !compressed && !sparse_solver
            data.jacobian_variables_dense[idx.equality, idx.variables] .= problem.equality_jacobian_variables # TODO
        elseif compressed && sparse_solver
            data.jacobian_variables_compressed_sparse.nzval .= problem.equality_jacobian_variables_compressed.nzval
        elseif compressed && !sparse_solver
            for (i, ii) in enumerate(idx.equality)
                for (j, jj) in enumerate(idx.equality)
                    data.jacobian_variables_compressed_dense[ii,jj] = problem.equality_jacobian_variables_compressed[ii,jj] # TODO
                end
            end
        end

        # cone product
        if !compressed && sparse_solver
            fill!(data.jacobian_variables_sparse, problem.cone_product_jacobian_duals, :cone_jacobian_duals)
            fill!(data.jacobian_variables_sparse, problem.cone_product_jacobian_slacks, :cone_jacobian_slacks)
        elseif !compressed && !sparse_solver
            data.jacobian_variables_dense[idx.cone_product, idx.duals] .= problem.cone_product_jacobian_duals # TODO
            data.jacobian_variables_dense[idx.cone_product, idx.slacks] .= problem.cone_product_jacobian_slacks # TODO
        end
    end

    if jacobian_parameters
        # jacobian parameters
        # equality
        if sparse_solver
            fill!(data.jacobian_parameters_sparse, problem.equality_jacobian_parameters, :equality_jacobian_parameters)
        else
            data.jacobian_parameters[idx.equality, idx.parameters] .= problem.equality_jacobian_parameters # TODO
            # data.jacobian_parameters.nzval .= problem.equality_jacobian_parameters.nzval
        end
    end
    return
end

function correction!(methods::ProblemMethods, data::SolverData, affine_step_size::Vector{T},
        step::Point{T}, step_correction::Point{T}, solution::Point{T}, central_path::Vector{T};
        complementarity_correction::T=0.5,
        compressed::Bool=false) where T

    num_cone = length(central_path)
    for i = 1:num_cone
        step_correction.duals[i] = complementarity_correction * affine_step_size[i] .* step.duals[i]
        step_correction.slacks[i] = complementarity_correction * affine_step_size[i] .* step.slacks[i]
    end

    correction!(methods, data, step_correction.duals, step_correction.slacks, solution, central_path, compressed=compressed)
    return
end

function correction!(methods::ProblemMethods, data::SolverData, Δz, Δs, solution::Point, κ; compressed::Bool=false)
    methods.correction(data.residual.all, data.residual.all, Δz, Δs, κ)
    if compressed
        methods.correction_compressed(data.residual_compressed.all, data.residual_compressed.all, Δz, Δs, solution.all, κ)
    end
    return nothing
end
