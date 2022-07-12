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
function residual!(data::SolverData, problem::ProblemData, idx::Indices, central_path;
        compressed::Bool=false,
        sparse_solver::Bool=false)

    # reset
    fill!(data.residual.all, 0.0)

    # Fill residual
    # equality
    data.residual.equality .= problem.equality_constraint
    # cone: z ∘ s - κ e
    for (i, ii) in enumerate(idx.cone_product)
        data.residual.all[ii] = problem.cone_product[i] - central_path[i] * problem.cone_target[i]
    end

    if compressed
        # Fill D
        equality_jacobian_variables = sparse_solver ? problem.equality_jacobian_variables_sparse : problem.equality_jacobian_variables
        for (i,ii) in enumerate(idx.slacks)
            data.slackness_jacobian_slacks .= problem.equality_jacobian_variables_sparse[idx.slackness[i], ii]
        end

        # compression corrections for the residual
        D = data.slackness_jacobian_slacks
        Zi = problem.cone_product_jacobian_inverse_slack_sparse
        data.cone_product_jacobian_inverse_slack .= Zi

        data.residual_compressed.all .= data.residual.all
        # data.residual_compressed.duals .-= D * Zi * data.residual.cone_product # - D Z⁻¹ cone_product
        mul!(data.point_temporary.duals, Zi, data.residual.cone_product) # - D Z⁻¹ cone_product
        data.point_temporary.duals .*= D # - D Z⁻¹ cone_product
        data.point_temporary.duals .*= -1.0 # - D Z⁻¹ cone_product
        for (i,ii) in enumerate(idx.duals)
            data.residual_compressed.duals[i] += data.point_temporary.duals[i] # - D Z⁻¹ cone_product
        end
        data.point_temporary.all .= 0.0
    end


    # Jacobian variables  
    # equality
    if !compressed && sparse_solver
        fill!(data.jacobian_parameters_sparse, problem.equality_jacobian_parameters_sparse, :equality_jacobian_parameters)
    elseif !compressed && !sparse_solver
        data.jacobian_parameters_dense[idx.equality, idx.parameters] .= problem.equality_jacobian_parameters # TODO
    elseif compressed && sparse_solver
        for (i, ii) in enumerate(idx.equality)
            for (j, jj) in enumerate(idx.equality)
                data.jacobian_variables_sparse_compressed[ii,jj] = problem.equality_jacobian_variables_sparse[ii,jj] # TODO
            end
        end
    elseif compressed && !sparse_solver
        for (i, ii) in enumerate(idx.equality)
            for (j, jj) in enumerate(idx.equality)
                data.jacobian_variables_dense_compressed[ii,jj] = problem.equality_jacobian_variables_dense[ii,jj] # TODO
            end
        end
    end
    
    # cone product
    if !compressed && sparse_solver
        fill!(data.jacobian_variables_sparse, problem.cone_product_jacobian_duals_sparse, :cone_jacobian_duals)
        fill!(data.jacobian_variables_sparse, problem.cone_product_jacobian_slacks_sparse, :cone_jacobian_slacks)
    elseif !compressed && !sparse_solver
        data.jacobian_variables[idx.cone_product, idx.duals] .= problem.cone_product_jacobian_duals_sparse # TODO
        data.jacobian_variables[idx.cone_product, idx.slacks] .= problem.cone_product_jacobian_slacks_sparse # TODO    
    elseif compressed
        # compression corrections for the jacobian
        D = data.slackness_jacobian_slacks
        Zi = problem.cone_product_jacobian_inverse_slack
        S = problem.cone_product_jacobian_duals
        data.cone_product_jacobian_inverse_slack .= Zi
        data.cone_product_jacobian_duals .= S

        # data.jacobian_variables_sparse_compressed[idx.duals, idx.duals] .+= -Zi * S # -Z⁻¹ S
        mul!(data.cone_product_jacobian_ratio, Zi, S) # -Z⁻¹ S
        data.cone_product_jacobian_ratio .*= -1.0 # -Z⁻¹ S
        for (i,ii) in enumerate(idx.duals)
            for (j,jj) in enumerate(idx.duals)
                if sparse_solver
                    data.jacobian_variables_sparse_compressed[ii,jj] += data.cone_product_jacobian_ratio[i,j] # -Z⁻¹ S
                else
                    data.jacobian_variables_dense_compressed[ii,jj] += data.cone_product_jacobian_ratio[i,j] # -Z⁻¹ S
                end
            end
        end
    end
    return
end

# n = 5
# A = rand(n,n)
# B = rand(n,n)
# C = rand(n,n)
# function mmy(A::Matrix{T}, B::Matrix{T}, C::Matrix{T}) where T
#     mul!(A, B, C)
#     return nothing
# end
#
# mmy(A, B, C)
# Main.@code_warntype mmy(A, B, C)
# @benchmark $mmy($A, $B, $C)
