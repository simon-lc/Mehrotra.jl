function residual!(data::SolverData228, problem::ProblemData228, idx::Indices228,
        solution::Point228, parameters, κ)
    x = solution.all
    y = solution.primals
    z = solution.duals
    s = solution.slacks
    θ = parameters

    # reset
    res = data.residual.all
    fill!(res, 0.0)

    # equality
    data.residual.equality .= problem.equality_constraint
    data.jacobian_variables[idx.equality, idx.variables] .= problem.equality_jacobian_variables # TODO
    data.jacobian_parameters[idx.equality, idx.parameters] .= problem.equality_jacobian_parameters # TODO

    # cone: z ∘ s - κ e
    for (i, ii) in enumerate(idx.cone_product)
        res[ii] = problem.cone_product[i] - κ[1] * problem.cone_target[i]
    end
    # Fill the jacobian
    data.jacobian_variables[idx.cone_product, idx.duals] .= problem.cone_product_jacobian_dual # TODO
    data.jacobian_variables[idx.cone_product, idx.slacks] .= problem.cone_product_jacobian_slack # TODO
    return
end



#
# function residual_symmetric!(residual_symmetric, residual, residual_second_order, matrix, idx::Indices)
#     # reset
#     fill!(residual_symmetric.all, 0.0)
#
#     rx = residual.variables
#     rr = residual.equality_slack
#     rs = residual.cone_slack
#     ry = residual.equality_dual
#     rz = residual.cone_dual
#     rt = residual.cone_slack_dual
#
#     for (i, rxi) in enumerate(rx)
#         residual_symmetric.variables[i] = rxi
#     end
#     for (i, ryi) in enumerate(ry)
#         residual_symmetric.equality[i] = ryi
#     end
#     for (i, rzi) in enumerate(rz)
#         residual_symmetric.cone[i] = rzi
#     end
#
#     # equality correction
#     for (i, ii) in enumerate(idx.symmetric_equality)
#         residual_symmetric.equality[i] += rr[i] / matrix[idx.equality_slack[i], idx.equality_slack[i]]
#     end
#
#     # cone correction (nonnegative)
#     for i in idx.cone_nonnegative
#         S̄i = matrix[idx.cone_slack_dual[i], idx.cone_slack_dual[i]]
#         Ti = matrix[idx.cone_slack_dual[i], idx.cone_slack[i]]
#         Pi = matrix[idx.cone_slack[i], idx.cone_slack[i]]
#         residual_symmetric.cone[i] += (rt[i] + S̄i * rs[i]) / (Ti + S̄i * Pi)
#     end
#
#     # cone correction (second-order)
#     for (i, idx_soc) in enumerate(idx.cone_second_order)
#         if !isempty(idx_soc)
#             C̄t = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack_dual[idx_soc]]
#             Cs = @views matrix[idx.cone_slack_dual[idx_soc], idx.cone_slack[idx_soc]]
#             P  = @views matrix[idx.cone_slack[idx_soc], idx.cone_slack[idx_soc]]
#             rs_soc = residual_second_order.cone_slack[i]
#             rt_soc = residual_second_order.cone_slack_dual[i]
#
#             residual_symmetric.cone[idx_soc] += second_order_matrix_inverse((Cs + C̄t * P), (C̄t * rs_soc + rt_soc))
#         end
#     end
#
#     return
# end