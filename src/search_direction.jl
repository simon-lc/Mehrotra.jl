
function search_direction_nonsymmetric!(step, data::SolverData228)
    # fill!(step, 0.0)
    step.all .= data.jacobian_variables \ data.residual.all
    step.all .*= -1.0
    return
end

# function search_direction!(solver)
#     # correct inertia
#     # inertia_correction!(solver)
#
#     # compute search direction
#     search_direction_symmetric!(solver.data.step, solver.data.residual, solver.data.jacobian_variables,
#         solver.data.step_symmetric, solver.data.residual_symmetric, solver.data.jacobian_variables_symmetric,
#         solver.indices,
#         solver.data.step_second_order, solver.data.residual_second_order,
#         solver.linear_solver;
#         update=solver.options.update_factorization)
#
#     # refine search direction
#     # solver.options.iterative_refinement && (!iterative_refinement!(solver.data.step, solver) && search_direction_nonsymmetric!(solver.data.step, solver.data))
# end
#
# function search_direction_symmetric!(
#     step::Point{T},
#     residual::Point{T},
#     matrix::SparseMatrixCSC{T,Int},
#     step_symmetric::PointSymmetric{T},
#     residual_symmetric::PointSymmetric{T},
#     matrix_symmetric::SparseMatrixCSC{T,Int},
#     idx::Indices,
#     step_second_order::SecondOrderViews{T},
#     residual_second_order::SecondOrderViews{T},
#     solver::LDLSolver{T,Int};
#     update=true) where T
#
#     # solve symmetric system
#     residual_symmetric!(residual_symmetric, residual, residual_second_order, matrix, idx)
#
#     linear_solve!(solver, step_symmetric.all, matrix_symmetric, residual_symmetric.all;
#         update=update)
#
#     # set Δx, Δy, Δz
#     Δx = step_symmetric.variables
#     Δy = step_symmetric.equality
#     Δz = step_symmetric.cone
#
#     for (i, Δxi) in enumerate(Δx)
#         step.variables[i] = Δxi
#     end
#     for (i, Δyi) in enumerate(Δy)
#         step.equality_dual[i] = Δyi
#     end
#     for (i, Δzi) in enumerate(Δz)
#         step.cone_dual[i] = Δzi
#     end
#
#     # recover Δr, Δs, Δt
#     Δr = step.equality_slack
#     Δs = step.cone_slack
#     Δt = step.cone_slack_dual
#     rr = residual.equality_slack
#     rs = residual.cone_slack
#     rt = residual.cone_slack_dual
#
#     # Δr
#     for (i, ii) in enumerate(idx.equality_slack)
#         Δr[i] = (rr[i] + Δy[i]) / matrix[idx.equality_slack[i], idx.equality_slack[i]]
#     end
#
#     # Δs, Δt (nonnegative)
#     for i in idx.cone_nonnegative
#         S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]]
#         Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
#         Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]]
#
#         Δs[i] = (rt[i] + S̄i * (rs[i] + Δz[i])) ./ (Ti + S̄i * Pi)
#         Δt[i] = (rt[i] - Ti * Δs[i]) / S̄i
#     end
#
#     # Δs, Δt (second-order)
#     for (i, idx_soc) in enumerate(idx.cone_second_order)
#         if !isempty(idx_soc)
#             C̄t = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]]
#             Cs = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
#             P  = @views matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]]
#
#             rs_soc = residual_second_order.cone_slack[i]
#             rt_soc = residual_second_order.cone_slack_dual[i]
#             Δz_soc = step_second_order.cone_dual[i]
#             Δs_soc = step_second_order.cone_slack[i]
#             Δt_soc = step_second_order.cone_slack_dual[i]
#
#             # Δs_soc .= (Cs + C̄t * P) \ (rt_soc + C̄t * (rs_soc + Δz_soc))
#             second_order_matrix_inverse!(Δs_soc, Cs + C̄t * P, rt_soc + C̄t * (rs_soc + Δz_soc))
#
#             # Δt_soc .= C̄t \ (rt_soc - Cs * Δs_soc)
#             second_order_matrix_inverse!(Δt_soc, C̄t, (rt_soc - Cs * Δs_soc))
#         end
#     end
#
#     return
# end
