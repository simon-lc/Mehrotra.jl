# function residual!(data::SolverData, problem::ProblemData, idx::Indices,
#         solution::Point, parameters, central_path)
#     x = solution.all
#     y = solution.primals
#     z = solution.duals
#     s = solution.slacks
#     θ = parameters
#
#     # reset
#     res = data.residual.all
#     fill!(res, 0.0)
#
#     # equality
#     data.residual.equality .= problem.equality_constraint
#     data.jacobian_variables[idx.equality, idx.variables] .= problem.equality_jacobian_variables # TODO
#     data.jacobian_parameters[idx.equality, idx.parameters] .= problem.equality_jacobian_parameters # TODO
#
#     # cone: z ∘ s - κ e
#     for (i, ii) in enumerate(idx.cone_product)
#         res[ii] = problem.cone_product[i] - central_path[i] * problem.cone_target[i]
#     end
#     # Fill the jacobian
#     data.jacobian_variables[idx.cone_product, idx.duals] .= problem.cone_product_jacobian_duals # TODO
#     data.jacobian_variables[idx.cone_product, idx.slacks] .= problem.cone_product_jacobian_slacks # TODO
#     return
# end


"""
    residual!(data, problem, idx, solution, parameters, central_path; compressed)
    compute non compressed residual and jacobian for problem structured as follows,
        jacobian_variables = [
            A B C
            D E F
            0 S Z
            ]
    compute compressed residual and jacobian for problem structured as follows,
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

    data: SolverData
    problem: ProblemData
    idx: Indices
    solution: Point
    parameters: Vector{T}
    central_path: Vector{T}
    compressed: Bool

"""
function residual!(data::SolverData, problem::ProblemData, idx::Indices,
        solution::Point, parameters, central_path; compressed::Bool=false)
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
    data.jacobian_parameters[idx.equality, idx.parameters] .= problem.equality_jacobian_parameters # TODO
    if compressed
        # data.compressed_jacobian_variables .= problem.equality_jacobian_variables[:, idx.equality] # TODO
        for (i, ii) in enumerate(idx.equality)
            for (j, jj) in enumerate(idx.equality)
                data.compressed_jacobian_variables[ii,jj] = problem.equality_jacobian_variables[ii,jj] # TODO
            end
        end
    else
        data.jacobian_variables[idx.equality, idx.variables] .= problem.equality_jacobian_variables # TODO
    end

    # cone: z ∘ s - κ e
    for (i, ii) in enumerate(idx.cone_product)
        res[ii] = problem.cone_product[i] - central_path[i] * problem.cone_target[i]
    end

    if compressed
        # compression corrections
        Zi = problem.cone_product_jacobian_inverse_slack
        S = problem.cone_product_jacobian_duals
        data.cone_product_jacobian_inverse_slack .= Zi
        data.cone_product_jacobian_duals .= S

        # data.compressed_jacobian_variables[idx.duals, idx.duals] .+= -Zi * S # -Z⁻¹ S
        mul!(data.cone_product_jacobian_ratio, Zi, S) # -Z⁻¹ S
        data.cone_product_jacobian_ratio .*= -1.0 # -Z⁻¹ S
        for (i,ii) in enumerate(idx.duals)
            for (j,jj) in enumerate(idx.duals)
                data.compressed_jacobian_variables[ii,jj] += data.cone_product_jacobian_ratio[i,j] # -Z⁻¹ S
            end
        end

        data.compressed_residual.all .= data.residual.all
        # data.compressed_residual.duals .-= Zi * data.residual.cone_product # - Z⁻¹ cone_product
        mul!(data.point_temporary.duals, Zi, data.residual.cone_product) # - Z⁻¹ cone_product
        data.point_temporary.duals .*= -1.0 # - Z⁻¹ cone_product
        for (i,ii) in enumerate(idx.duals)
            data.compressed_residual.duals[i] += data.point_temporary.duals[i] # - Z⁻¹ cone_product
        end
        data.point_temporary.all .= 0.0
    else
        # Fill the jacobian
        data.jacobian_variables[idx.cone_product, idx.duals] .= problem.cone_product_jacobian_duals # TODO
        data.jacobian_variables[idx.cone_product, idx.slacks] .= problem.cone_product_jacobian_slacks # TODO
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
