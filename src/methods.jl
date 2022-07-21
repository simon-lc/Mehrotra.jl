abstract type AbstractProblemMethods{T,E,EC,EX,EXC,EP,C,CC,S}
end

struct ProblemMethods{T,E,EC,EX,EXC,EP,C,CC,S} <: AbstractProblemMethods{T,E,EC,EX,EXC,EP,C,CC,S}
    equality_constraint::E                             # e
    equality_constraint_compressed::EC                 # ec
    equality_jacobian_variables::EX                    # ex
    equality_jacobian_variables_compressed::EXC        # exc
    equality_jacobian_parameters::EP                   # eθ
    correction::C                                      # c
    correction_compressed::CC                          # cc
    slack_direction::S                                 # s
    equality_jacobian_variables_cache::Vector{T}
    equality_jacobian_variables_compressed_cache::Vector{T}
    equality_jacobian_parameters_cache::Vector{T}
    equality_jacobian_variables_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_variables_compressed_sparsity::Vector{Tuple{Int,Int}}
    equality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
end

function symbolics_methods(equality::Function, dim::Dimensions, idx::Indices)
    e, ec, ex, exc, eθ, c, cc, s, ex_sparsity, exc_sparsity, eθ_sparsity = generate_full_gradients(equality, dim, idx)

    methods = ProblemMethods(
        e,
        ec,
        ex,
        exc,
        eθ,
        c,
        cc,
        s,
        zeros(length(ex_sparsity)),
        zeros(length(exc_sparsity)),
        zeros(length(eθ_sparsity)),
        ex_sparsity,
        exc_sparsity,
        eθ_sparsity,
    )

    return methods
end

# """
#     StruturedProblemMethods112{T,E,EX,EP} <: AbstractProblemMethods{T,E,EX,EP}
# 	Store fast residual and Jacobian methods for structured NCPs. These structured NCPs include:
# 		- linear programs (LP)
# 		- quadraric programs (QP)
# 		- nonlinear complementarity program arizing from contact formulation (Contact NCP and LCP)
# 	We can compress the Jacobian by solving for primals and duals only (slacks are computed from the primals and duals).

# 	For LPs, QPs and frictionless contct NCPs, the compressed Jacobian matrix is symmetric.
# 	In this case, we can leverage QDLDL to factorize the compressed matrix efficiently.
# 	Otherwise we need to devise a QDLDL-inspired sparse solver.

# 	Non-structured programs look like this,
#         jacobian_variables = [
#             A B C
#             D E F
#             0 S Z
#             ]
# 	Structured programs look like this,
#         jacobian_variables = [
#             A B 0
#             C 0 D
#             0 S Z
#             ]
#     where
#         A = ∂optimality_constraints / ∂y == num_primals, num_primals
#         B = ∂optimality_constraints / ∂z == num_primals, num_cone
#         C = ∂slackness_constraints / ∂y == num_cone, num_primals
#         D = ∂slackness_constraints / ∂s = diagonal matrix == num_cone, num_cone
#         S = ∂(z∘s) / ∂z == num_cone, num_cone
#         Z = ∂(z∘s) / ∂s == num_cone, num_cone

#     we solve
#         |A B 0| |Δy|   |-optimality    |
#         |C 0 D|×|Δz| = |-slack_equality|
#         |0 S Z| |Δs|   |-cone_product  |
#     we get the compressed form
#         |A B      | |Δy|   |-optimality                        |
#         |C -DZ⁻¹S |×|Δz| = |-slack_equality + DZ⁻¹ cone_product|
#         Δs = -Z⁻¹ (cone_product + S * Δz)
# """
# struct StructuredProblemMethods112{T,O,OY,OZ,OP,S,SY,SS,SP} <: AbstractProblemMethods{T,O,OY,OZ}
#     optimality_constraint::O                             # o
#     optimality_jacobian_primals::OY                      # oy
#     optimality_jacobian_duals::OZ                        # oz
#     optimality_jacobian_parameters::OP                   # oθ

#     slackness_constraint::S                              # s
#     slackness_jacobian_primals::SY                       # sy
#     slackness_jacobian_slacks::SS                        # ss
#     slackness_jacobian_parameters::SP                    # sθ

#     optimality_jacobian_primals_cache::Vector{T}
#     optimality_jacobian_duals_cache::Vector{T}
#     optimality_jacobian_parameters_cache::Vector{T}
#     optimality_jacobian_primals_sparsity::Vector{Tuple{Int,Int}}
#     optimality_jacobian_duals_sparsity::Vector{Tuple{Int,Int}}
#     optimality_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}

#     slackness_jacobian_primals_cache::Vector{T}
#     slackness_jacobian_slacks_cache::Vector{T}
#     slackness_jacobian_parameters_cache::Vector{T}
#     slackness_jacobian_primals_sparsity::Vector{Tuple{Int,Int}}
#     slackness_jacobian_slacks_sparsity::Vector{Tuple{Int,Int}}
#     slackness_jacobian_parameters_sparsity::Vector{Tuple{Int,Int}}
# end

# function structured_symbolics_methods(equality::Function, dim::Dimensions, idx::Indices)
#     o, oy, oz, oθ, s, sy, ss, sθ, oy_sp, oz_sp, oθ_sp, sy_sp, ss_sp, sθ_sp = generate_structured_gradients(equality, dim, idx)

#     methods = StructuredProblemMethods112(
#         o,
#         oy,
#         oz,
#         oθ,
#         s,
#         sy,
#         ss,
#         sθ,
#         zeros(length(oy_sp)),
#         zeros(length(oz_sp)),
#         zeros(length(oθ_sp)),
#         oy_sp,
#         oz_sp,
#         oθ_sp,
#         zeros(length(sy_sp)),
#         zeros(length(ss_sp)),
#         zeros(length(sθ_sp)),
#         sy_sp,
#         ss_sp,
#         sθ_sp,
#     )

#     return methods
# end
