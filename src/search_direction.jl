"""
    search_direction!(solver)

    compute search direction for structured residual jacobians,
        jacobian_variables = [
            A B 0
            C 0 I
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
        |C 0 I|×|Δz| = |-slack_equality|
        |0 S Z| |Δs|   |-cone_product  |
    we get the compressed form
        |A B     | |Δy|   |-optimality                       |
        |C -Z⁻¹S |×|Δz| = |-slack_equality + Z⁻¹ cone_product|
        Δs = -Z⁻¹ (cone_product + S * Δz)

    solver: Solver228
"""
function search_direction!(solver::Solver228)
    dimensions = solver.dimensions
    linear_solver = solver.linear_solver
    data = solver.data
    step = data.step
    compressed = solver.options.compressed_search_direction

    if compressed
        compressed_search_direction!(linear_solver, dimensions, data, step)
    else
        uncompressed_search_direction!(linear_solver, dimensions, data, step)
    end
    return nothing
end

function compressed_search_direction!(linear_solver::LUSolver{T},
        dimensions::Dimensions228,
        data::SolverData228{T},
        step::Point228{T},
        ) where T


    Zi = data.cone_product_jacobian_inverse_slack
    S = data.cone_product_jacobian_dual
    # primal dual step
    # step.equality .= data.compressed_jacobian_variables \ residual.equality
    data.dense_compressed_jacobian_variables .= data.compressed_jacobian_variables

    linear_solve!(linear_solver, step.equality,
        data.dense_compressed_jacobian_variables, data.compressed_residual.equality)
    step.equality .*= -1.0

    # slack step
    # step.slacks .= -Zi * (data.compressed_residual.cone_product + S * step.duals) # -Z⁻¹ (cone_product + S * Δz)
    data.point_temporary.slacks .= 0.0
    mul!(data.point_temporary.slacks, S, step.duals)
    data.point_temporary.slacks .+= data.compressed_residual.cone_product
    mul!(step.slacks, Zi, data.point_temporary.slacks)
    step.slacks .*= -1.0
    data.point_temporary.slacks .= 0.0
    return nothing
end


function uncompressed_search_direction!(linear_solver::LUSolver{T},
        dimensions::Dimensions228,
        data::SolverData228{T},
        step::Point228{T},
        ) where T


    data.dense_jacobian_variables .= data.jacobian_variables
    # ny = dimensions.primals
    # nz = dimensions.duals
    # ns = dimensions.slacks

    linear_solve!(
        linear_solver,
        step.all,
        # data.dense_jacobian_variables + Diagonal([1.0*1e-8ones(ny); zeros(nz+ns)]),
        data.dense_jacobian_variables,
        data.residual.all,
        fact=true)
    step.all .*= -1.0
    return nothing
end


# step0 = solver.data.step
# data = solver.data
# residual = solver.data.residual
# linear_solve!(solver.linear_solver, step0.equality,
#     data.dense_compressed_jacobian_variables, residual.equality)
#
#
# step0.all[solver.indices.duals]

#
#
# solver.solution.all
# z = solver.solution.duals
# s = solver.solution.slacks
# idx.
# z = ones(100)
# s = ones(100)
# z = [i+1.0 for i=1:100]
# s = [-i-1.0 for i=1:100]
# solver.cone_methods.product(z, s, idx_nn, idx_soc)
# solver.cone_methods.product_jacobian(z, s, idx_nn, idx_soc)
# solver.cone_methods.product_jacobian_inverse(z, s, idx_nn, idx_soc)
#
# z = ones(10)
# s = ones(10)
# cone_product_jacobian(z, s, idx_nn, idx_soc)
# cone_product_jacobian_inverse(z, s, idx_nn, idx_soc)
#
# solver.data
#
#
#
# structured_search_direction!(solver)
# solver.data.step.equality
#
# solver.data
# solver.problem
#
# # function search_direction!(solver)
# #     # correct inertia
# #     # inertia_correction!(solver)
# #
# #     # compute search direction
# #     search_direction_symmetric!(solver.data.step, solver.data.residual, solver.data.jacobian_variables,
# #         solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.jacobian_variables_symmetric,
# #         solver.indices,
# #         solver.data.step_second_order, solver.data.residual_second_order,
# #         solver.linear_solver;
# #         update=solver.options.update_factorization)
# #
# #     # refine search direction
# #     # solver.options.iterative_refinement && (!iterative_refinement!(solver.data.step, solver) && search_direction_nonsymmetric!(solver.data.step, solver.data))
# # end
# #
# # function search_direction_symmetric!(
# #     step::Point{T},
# #     residual::Point{T},
# #     matrix::SparseMatrixCSC{T,Int},
# #     step_symmetric::PointSymmetric{T},
# #     residual_symmetric::PointSymmetric{T},
# #     matrix_symmetric::SparseMatrixCSC{T,Int},
# #     idx::Indices,
# #     step_second_order::SecondOrderViews{T},
# #     residual_second_order::SecondOrderViews{T},
# #     solver::LDLSolver{T,Int};
# #     update=true) where T
# #
# #     # solve symmetric system
# #     residual_symmetric!(residual_symmetric, residual, residual_second_order, matrix, idx)
# #
# #     linear_solve!(solver, step_symmetric.all, matrix_symmetric, residual_symmetric.all;
# #         update=update)
# #
# #     # set Δx, Δy, Δz
# #     Δx = step_symmetric.variables
# #     Δy = step_symmetric.equality
# #     Δz = step_symmetric.cone
# #
# #     for (i, Δxi) in enumerate(Δx)
# #         step.variables[i] = Δxi
# #     end
# #     for (i, Δyi) in enumerate(Δy)
# #         step.equality_dual[i] = Δyi
# #     end
# #     for (i, Δzi) in enumerate(Δz)
# #         step.cone_dual[i] = Δzi
# #     end
# #
# #     # recover Δr, Δs, Δt
# #     Δr = step.equality_slack
# #     Δs = step.cone_slack
# #     Δt = step.cone_slack_dual
# #     rr = residual.equality_slack
# #     rs = residual.cone_slack
# #     rt = residual.cone_slack_dual
# #
# #     # Δr
# #     for (i, ii) in enumerate(idx.equality_slack)
# #         Δr[i] = (rr[i] + Δy[i]) / matrix[idx.equality_slack[i], idx.equality_slack[i]]
# #     end
# #
# #     # Δs, Δt (nonnegative)
# #     for i in idx.cone_nonnegative
# #         S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]]
# #         Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
# #         Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]]
# #
# #         Δs[i] = (rt[i] + S̄i * (rs[i] + Δz[i])) ./ (Ti + S̄i * Pi)
# #         Δt[i] = (rt[i] - Ti * Δs[i]) / S̄i
# #     end
# #
# #     # Δs, Δt (second-order)
# #     for (i, idx_soc) in enumerate(idx.cone_second_order)
# #         if !isempty(idx_soc)
# #             C̄t = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]]
# #             Cs = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
# #             P  = @views matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]]
# #
# #             rs_soc = residual_second_order.cone_slack[i]
# #             rt_soc = residual_second_order.cone_slack_dual[i]
# #             Δz_soc = step_second_order.cone_dual[i]
# #             Δs_soc = step_second_order.cone_slack[i]
# #             Δt_soc = step_second_order.cone_slack_dual[i]
# #
# #             # Δs_soc .= (Cs + C̄t * P) \ (rt_soc + C̄t * (rs_soc + Δz_soc))
# #             second_order_matrix_inverse!(Δs_soc, Cs + C̄t * P, rt_soc + C̄t * (rs_soc + Δz_soc))
# #
# #             # Δt_soc .= C̄t \ (rt_soc - Cs * Δs_soc)
# #             second_order_matrix_inverse!(Δt_soc, C̄t, (rt_soc - Cs * Δs_soc))
# #         end
# #     end
# #
# #     return
# # end
# solver.cone_methods
# solver.indices.
