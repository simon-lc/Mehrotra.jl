abstract type LinearSolver end


"""
    Empty solver
"""
mutable struct EmptySolver <: LinearSolver
    F::Any
end

function empty_solver(A::Any)
    EmptySolver(A)
end

# QDLDL take the one form CI-MPC
# QDLDL.QDLDL_factor! is causing some allocations


"""
    LU solver
"""
mutable struct LUSolver{T} <: LinearSolver
    A::Array{T,2}
    x::Vector{T}
    ipiv::Vector{Int}
    lda::Int
    info::Base.RefValue{Int}
end

function lu_solver(A)
    m, n = size(A)
    x = zeros(m)
    ipiv = similar(A, LinearAlgebra.BlasInt, min(m, n))
    lda  = max(1, stride(A, 2))
    info = Ref{LinearAlgebra.BlasInt}()
    LUSolver(copy(A), x, ipiv, lda, info)
end

function getrf!(A, ipiv, lda, info)
    Base.require_one_based_indexing(A)
    LinearAlgebra.chkstride1(A)
    m, n = size(A)
    lda  = max(1,stride(A, 2))
    ccall((LinearAlgebra.BLAS.@blasfunc(dgetrf_), Base.liblapack_name), Cvoid,
          (Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64},
           Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}),
          m, n, A, lda, ipiv, info)
    return nothing
end

function factorize!(s::LUSolver{T}, A::AbstractMatrix{T}) where T
    fill!(s.A, 0.0)
    fill!(s.ipiv, 0)
    s.lda = 0
    s.A .= A
    getrf!(s.A, s.ipiv, s.lda, s.info)
end

function linear_solve!(s::LUSolver{T}, x::AbstractVector{T}, A::Matrix{T},
        b::AbstractVector{T}; reg::T = 0.0, fact::Bool = true) where T
    fact && factorize!(s, A)
    s.x .= b
    LinearAlgebra.LAPACK.getrs!('N', s.A, s.ipiv, s.x)
    x .= s.x
    return nothing
end

function linear_solve!(s::LUSolver{T}, x::AbstractMatrix{T}, A::Matrix{T},
    b::AbstractMatrix{T}; reg::T = 0.0, fact::Bool = true) where T
    fill!(x, 0.0)
    n, m = size(x)
    r_idx = 1:n
    fact && factorize!(s, A)
    x .= b
    for j = 1:m
        xv = @views x[r_idx, j]
        LinearAlgebra.LAPACK.getrs!('N', s.A, s.ipiv, xv)
    end
end

"""
    Sparse LU solver
"""
mutable struct SparseLUSolver113{T} <: LinearSolver
    x::Vector{T}
    # factorization::ILU0Precon{T,Int,T}
    factorization::ILU0Precon{T,Int,T}
end

function sparse_lu_solver(A)
    m, n = size(A)
    x = zeros(m)
    factorization = ilu0(A)
    SparseLUSolver113(x, factorization)
end

function factorize!(s::SparseLUSolver113{T}, A::SparseMatrixCSC{T,Int}) where T
    ilu0!(s.factorization, A)
end

function linear_solve!(s::SparseLUSolver113{T}, x::AbstractVector{T}, A::SparseMatrixCSC{T,Int},
        b::AbstractVector{T}; reg::T = 0.0, fact::Bool = true) where T
    fact && factorize!(s, A)
    ldiv!(x, s.factorization, b)
    return nothing
end

function linear_solve!(s::SparseLUSolver113{T}, x::AbstractMatrix{T}, A::SparseMatrixCSC{T,Int},
    b::AbstractMatrix{T}; reg::T = 0.0, fact::Bool = true) where T
    ldiv!(x, s.factorization, b)
end



n = 5
A = sprand(n, n, 0.6)# + 10*I
cond(Matrix(A))
lu_factorization = ilu0(A)
x = zeros(n)
b = rand(n)
ldiv!(x, lu_factorization, deepcopy(b))
norm(A * x - b)
A * x - b

# x = lu_factorization \ b
# A * x - b
#

B = sprand(10,10,0.6) - 1e-1*I
B = B+B'
Bc = deepcopy(B)
lu_factorization = ilu0(B)
ilu0!(lu_factorization, B)
x = zeros(10)
r = rand(10)
rc = deepcopy(r)
ldiv!(x, lu_factorization, r)
norm(Bc * x - rc, Inf)
ldiv!(x, lu(B), r)
norm(Bc * x - rc, Inf)




B = sprand(10,10,1.0) # works
B = sprand(10,10,0.2) # doesn't work
lu_factorization = ilu0(B)
x = zeros(10)
r = rand(10)
ldiv!(x, lu_factorization, r)
norm(B * x - r, Inf)
ldiv!(x, lu(B), r)
norm(B * x - r, Inf)



# n = 5
# m = 3
# A = rand(n,n)
# A = A'*A
# linear_solver = lu_solver(A)
#
# X = rand(n,m)
# B = rand(n,m)
# Xv = view(X, Vector(1:n), Vector(1:2))
# Bv = view(B, Vector(1:n), Vector(1:2))
# # Xv = view(X, 1:n, 1:2)
# # Bv = view(B, 1:n, 1:2)
# my_linear_solve!(linear_solver, X, A, B)
# my_linear_solve!(linear_solver, Xv, A, Bv)
#







#
# fieldnames(typeof(lu(A)))
# lu(A).symbolic
# lu(A).numeric
# lu(A).m
# lu(A).n
# lu(A).colptr
# lu(A).rowval
# lu(A).nzval
# lu(A).status
#
# lu(A).Rs
# lu(A).L
# lu(A).U
#
# fact = lu(A)
# lu!(fact, A)
# @benchmark $lu!($fact, $A)
# @benchmark $lu($A)
#
# using SuiteSparse
# using SparseArrays
# import SparseArrays.decrement
# using LinearAlgebra
# import LinearAlgebra.lu!
# import ..decrement
#
# function mylu1!(F::SuiteSparse.UMFPACK.UmfpackLU, S::SparseMatrixCSC{T,I}; check::Bool=true) where {T,I}
#     zerobased = SparseArrays.getcolptr(S)[1] == 0
#     # resize workspace if needed
#     if F.n < size(S, 2)
#         F.workspace = SparseArrays.UmfpackWS(S)
#     end
#
#     F.m = size(S, 1)
#     F.n = size(S, 2)
#     F.colptr = zerobased ? copy(SparseArrays.getcolptr(S)) : decrement(SparseArrays.getcolptr(S))
#     F.rowval = zerobased ? copy(rowvals(S)) : decrement(rowvals(S))
#     F.nzval = copy(nonzeros(S))
#
#     # SuiteSparse.UMFPACK.umfpack_numeric!(F, reuse_numeric = false)
#     SuiteSparse.UMFPACK.umfpack_numeric!(F, reuse_numeric = true)
#     check && (issuccess(F) || throw(LinearAlgebra.SingularException(0)))
#     return F
# end
#
# # Convert from 1-based to 0-based indices
# function decrement!(A::AbstractArray{T}) where T<:Integer
#     for i in eachindex(A); A[i] -= oneunit(T) end
#     A
# end
# decrement(A::AbstractArray{<:Integer}) = decrement!(copy(A))
#
# n = 19
# A = sprand(n, n, 1.0)
# AA = 2*A
# factA = lu(A)
# factAA = mylu1!(factA, AA)
# @benchmark $mylu1!($factA, $AA)
# r = rand(n)
# x = zeros(n)
# ldiv!(x, factAA, r)
# AA * x - r
# A * x - r
#
# SparseArrays.getcolptr(A)[1]







#
#
# """
#     QDLDL inplace functionality
# """
# mutable struct LDLSolver{Tf<:AbstractFloat,Ti<:Integer} <: LinearSolver
#     # QDLDL Factorization
#     F::QDLDL.QDLDLFactorisation{Tf,Ti}
#     A_sparse::SparseMatrixCSC{Tf,Ti}
#     # Allocate memory
#     AtoPAPt::Vector{Ti}
#     Pr::Vector{Ti}
#     Pc::Vector{Ti}
#     Pv::Vector{Tf}
#     num_entries::Vector{Ti}
#     Pr_triu::Vector{Ti}
#     Pv_triu::Vector{Tf}
# end
#
# function LDLSolver(A::SparseMatrixCSC{Tf,Ti}, F::QDLDL.QDLDLFactorisation{Tf,Ti}) where {Tf<:AbstractFloat,Ti<:Integer}
#     AtoPAPt = zeros(Ti,nnz(A))
#     Pr = zeros(Ti, nnz(A))
#     Pc = zeros(Ti, size(A, 1) + 1)
#     Pv = zeros(Tf, nnz(A))
#     num_entries = zeros(Ti,size(A, 2))
#     Pr_triu = zeros(Ti, nnz(F.workspace.triuA))
#     Pv_triu = zeros(Tf, nnz(F.workspace.triuA))
#     return LDLSolver{Tf,Ti}(F, copy(A), AtoPAPt, Pr, Pc, Pv, num_entries, Pr_triu, Pv_triu)
# end
#
# function factorize!(s::LDLSolver{Tf,Ti}, A::SparseMatrixCSC{Tf,Ti}) where {Tf<:AbstractFloat, Ti<:Integer}
#     # Reset the pre-allocated fields
#     s.AtoPAPt .= 0
# 	s.Pr .= 0
#     s.Pc .= 0
# 	s.Pv .= 0.0
#     s.num_entries .= 0
# 	s.Pr_triu .= 0
# 	s.Pv_triu .= 0.0
#
#     # Triangularize the matrix with the allocation-free method.
#     A = _permute_symmetricAF(A, s.AtoPAPt, s.F.iperm, s.Pr, s.Pc, s.Pv,
# 		s.num_entries, s.Pr_triu, s.Pv_triu)  #returns an upper triangular matrix
#
#     # Update the workspace, triuA is the only field we need to update
#     s.F.workspace.triuA.nzval .= A.nzval
#
#     # factor the matrix
#     QDLDL.refactor!(s.F)
#
#     # return nothing
# end
#
# # the main function without extra argument checks
# # following the book: Timothy Davis - Direct Methods for Sparse Linear Systems
# function _permute_symmetricAF(
#         A::SparseMatrixCSC{Tf, Ti},
#         AtoPAPt::AbstractVector{Ti},
#         iperm::AbstractVector{Ti},
# 		Pr::AbstractVector{Ti},
#         Pc::AbstractVector{Ti},
#         Pv::AbstractVector{Tf},
# 		num_entries::Vector{Ti},
# 		Pr_triu::AbstractVector{Ti},
# 		Pv_triu::AbstractVector{Tf},
#         ) where {Tf <: AbstractFloat, Ti <: Integer}
#
#     # 1. count number of entries that each column of P will have
#     n = size(A, 2)
#     Ar = A.rowval
#     Ac = A.colptr
#     Av = A.nzval
#
#     # count the number of upper-triangle entries in columns of P, keeping in mind the row permutation
#     for colA = 1:n
#         colP = iperm[colA]
#         # loop over entries of A in column A...
#         for row_idx = Ac[colA]:Ac[colA+1]-1
#             rowA = Ar[row_idx]
#             rowP = iperm[rowA]
#             # ...and check if entry is upper triangular
#             if rowA <= colA
#                 # determine to which column the entry belongs after permutation
#                 col_idx = max(rowP, colP)
#                 num_entries[col_idx] += one(Ti)
#             end
#         end
#     end
#     # 2. calculate permuted Pc = P.colptr from number of entries
#     Pc[1] = one(Ti)
#     @inbounds for k = 1:n
#         Pc[k + 1] = Pc[k] + num_entries[k]
#
#         # reuse this vector memory to keep track of free entries in rowval
#         num_entries[k] = Pc[k]
#     end
#     # use alias
#     row_starts = num_entries
#
#     # 3. permute the row entries and position of corresponding nzval
#     for colA = 1:n
#         colP = iperm[colA]
#         # loop over rows of A and determine where each row entry of A should be stored
#         for rowA_idx = Ac[colA]:Ac[colA+1]-1
#             rowA = Ar[rowA_idx]
#             # check if upper triangular
#             if rowA <= colA
#                 rowP = iperm[rowA]
#                 # determine column to store the entry
#                 col_idx = max(colP, rowP)
#
#                 # find next free location in rowval (this results in unordered columns in the rowval)
#                 rowP_idx = row_starts[col_idx]
#
#                 # store rowval and nzval
#                 Pr[rowP_idx] = min(colP, rowP)
#                 Pv[rowP_idx] = Av[rowA_idx]
#
#                 #record this into the mapping vector
#                 AtoPAPt[rowA_idx] = rowP_idx
#
#                 # increment next free location
#                 row_starts[col_idx] += 1
#             end
#         end
#     end
# 	nz_new = Pc[end] - 1
# 	for i = 1:nz_new
# 		Pr_triu[i] = Pr[i]
# 		Pv_triu[i] = Pv[i]
# 	end
#     P = SparseMatrixCSC{Tf, Ti}(n, n, Pc, Pr_triu, Pv_triu)
#
#     return P
# end
#
# """
#     LDL solver
# """
# function ldl_solver(A::SparseMatrixCSC{T,Int}) where T
#     LDLSolver(A, qdldl(A))
# end
#
# function linear_solve4!(solver::LDLSolver{Tv,Ti}, x::Vector{Tv}, A::SparseMatrixCSC{Tv,Ti}, b::Vector{Tv};
#     reg=0.0, fact::Bool = true) where {Tv<:AbstractFloat,Ti<:Integer}
#     # fact && factorize!(solver, A) # factorize
#     # x .= b
#     # QDLDL.solve!(solver.F, x) # solve
# end
#
# function linear_solve3!(s::LDLSolver{T}, x::Matrix{T}, A::Matrix{T},
#     b::Matrix{T};
#     reg::T = 0.0,
#     fact::Bool = true) where T
#
#     fill!(x, 0.0)
#     n, m = size(x)
#     r_idx = 1:n
#     fact && factorize!(s, A)
#
#     x .= b
#     for j = 1:m
#         xv = @views x[r_idx, j]
#         QDLDL.solve!(solver.F, xv)
#     end
# end
#
# function linear_solve2!(solver::LDLSolver{Tv,Ti}, x::Vector{Tv},
# 		A::SparseMatrixCSC{Tv,Ti}, b::Vector{Tv};
# 	    reg=0.0,
# 		fact::Bool=true) where {Tv<:AbstractFloat,Ti<:Integer}
#
#     # fill sparse_matrix
#     n, m = size(A)
#     for i = 1:n
#         for j = 1:m
#             solver.A_sparse[i, j] = A[i, j]
#         end
#     end
#
#     linear_solve3!(solver, x, solver.A_sparse, b, reg=reg, fact=fact)
# end
#
#
# function linear_solve4!(solver::LDLSolver{Tv,Ti}, x::Vector{Tv}, A::SparseMatrixCSC{Tv,Ti}, b::Vector{Tv};
#     reg=0.0, fact::Bool = true) where {Tv<:AbstractFloat,Ti<:Integer}
#     fact && factorize!(solver, A) # factorize
#     x .= b
#     QDLDL.solve!(solver.F, x) # solve
# end
#
#
# n = 500
# Ap = rand(24)
# Am = rand(n-24)
# A = sparse(cat(Ap*Ap'+I, -Am*Am'-I, dims=(1,2)))
# solver = ldl_solver(A)
#
# x = zeros(n)
# b = rand(n)
# linear_solve4!(solver, x, A, b)
# norm(A * x - b, Inf)
#
# Main.@code_warntype linear_solve4!(solver, x, A, b)
#
#
# function factorize!(s::LDLSolver{Tf,Ti}, A::SparseMatrixCSC{Tf,Ti}) where {Tf<:AbstractFloat, Ti<:Integer}
#     # Reset the pre-allocated fields
#     s.AtoPAPt .= 0
# 	s.Pr .= 0
#     s.Pc .= 0
# 	s.Pv .= 0.0
#     s.num_entries .= 0
# 	s.Pr_triu .= 0
# 	s.Pv_triu .= 0.0
#
#     # Triangularize the matrix with the allocation-free method.
#     A = _permute_symmetricAF(A, s.AtoPAPt, s.F.iperm, s.Pr, s.Pc, s.Pv,
# 		s.num_entries, s.Pr_triu, s.Pv_triu)  #returns an upper triangular matrix
#
#     # Update the workspace, triuA is the only field we need to update
#     s.F.workspace.triuA.nzval .= A.nzval
#
#     # factor the matrix
#     QDLDL.refactor!(s.F)
#
#     return nothing
# end
#
#
# function factor!(workspace::QDLDL.QDLDLWorkspace{Tf,Ti},logical::Bool) where {Tf<:AbstractFloat,Ti<:Integer}
#
#     if (logical)
#         workspace.Lx   .= 1
#         workspace.D    .= 1
#         workspace.Dinv .= 1
#     end
#
#     #factor using QDLDL converted code
#     A = workspace.triuA
#     posDCount  = QDLDL.QDLDL_factor!(A.n,A.colptr,A.rowval,A.nzval,
#                               workspace.Lp,
#                               workspace.Li,
#                               workspace.Lx,
#                               workspace.D,
#                               workspace.Dinv,
#                               workspace.Lnz,
#                               workspace.etree,
#                               workspace.bwork,
#                               workspace.iwork,
#                               workspace.fwork,
#                               logical,
#                               workspace.Dsigns,
#                               workspace.regularize_eps,
#                               workspace.regularize_delta,
#                               workspace.regularize_count
#                               )
#
#     # if (posDCount < 0)
#     #     error("Zero entry in D (matrix is not quasidefinite)")
#     # end
#
#     # workspace.positive_inertia[] = posDCount
#
#     return nothing
#
# end
#
#
# # the issue is within QDLDL.QDLDL_factor!(
#
# solver.F
#
# A
# factorize!(solver, A)
# refactor!(solver.F)
# @benchmark $factor!($solver.F.workspace, false)
# @benchmark $refactor!($solver.F)
# @benchmark $factorize!($solver, $A)
# # @benchmark $linear_solve!($A)
# # @benchmark $linear_solve!($x)
# # @benchmark $linear_solve!($x, $b)
# # @benchmark $linear_solve2!($x, $A, $b)
# @benchmark $linear_solve4!($solver, $x, $A, $b)
#
# a = 10
# a = 10
# a = 10
# a = 10
#
#
#
# """
#     LU solver
# """
# mutable struct LUSolver{T} <: LinearSolver
#     A::Array{T,2}
#     ipiv::Vector{Int}
#     lda::Int
#     info::Base.RefValue{Int}
# end
#
# function lu_solver(A)
#     m, n = size(A)
#     ipiv = similar(A, LinearAlgebra.BlasInt, min(m, n))
#     lda  = max(1, stride(A, 2))
#     info = Ref{LinearAlgebra.BlasInt}()
#     LUSolver(copy(A), ipiv, lda, info)
# end
#
# function getrf!(A, ipiv, lda, info)
#     Base.require_one_based_indexing(A)
#     LinearAlgebra.chkstride1(A)
#     m, n = size(A)
#     lda  = max(1,stride(A, 2))
#     ccall((LinearAlgebra.BLAS.@blasfunc(dgetrf_), Base.liblapack_name), Cvoid,
#           (Ref{LinearAlgebra.BlasInt}, Ref{LinearAlgebra.BlasInt}, Ptr{Float64},
#            Ref{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}, Ptr{LinearAlgebra.BlasInt}),
#           m, n, A, lda, ipiv, info)
#     return nothing
# end
#
# function factorize!(s::LUSolver{T}, A::AbstractMatrix{T}) where T
#     fill!(s.A, 0.0)
#     fill!(s.ipiv, 0)
#     s.lda = 0
#     s.A .= A
#     getrf!(s.A, s.ipiv, s.lda, s.info)
# end
#
# function linear_solve!(s::LUSolver{T}, x::Vector{T}, A::Matrix{T},
#         b::Vector{T}; reg::T = 0.0, fact::Bool = true) where T
#     fact && factorize!(s, A)
#     x .= b
#     LinearAlgebra.LAPACK.getrs!('N', s.A, s.ipiv, x)
# end
#
# function linear_solve!(s::LUSolver{T}, x::Matrix{T}, A::Matrix{T},
#     b::Matrix{T}; reg::T = 0.0, fact::Bool = true) where T
#     fill!(x, 0.0)
#     n, m = size(x)
#     r_idx = 1:n
#     fact && factorize!(s, A)
#     x .= b
#     for j = 1:m
#         xv = @views x[r_idx, j]
#         LinearAlgebra.LAPACK.getrs!('N', s.A, s.ipiv, xv)
#     end
# end
#
#
#
#
# # function linear_solve!(solver::LDLSolver111{Tv,Ti}, x::Vector{Tv}, A::AbstractMatrix{Tv}, b::Vector{Tv};
# #     fact=true,
# #     update=true) where {Tv<:AbstractFloat,Ti<:Integer}
#
# #     # fill sparse_matrix
# #     n, m = size(A)
# #     for i = 1:n
# #         for j = 1:m
# #             solver.A_sparse[i, j] = A[i, j]
# #         end
# #     end
#
# #     linear_solve!(solver, x, solver.A_sparse, b,
# #         fact=fact,
# #         update=update)
# # end
#
# function linear_solve!(s::LDLSolver111{T}, x::Matrix{T}, A::Matrix{T},
#     b::Matrix{T};
#     fact=true,
#     update=true) where T
#
#     fill!(x, 0.0)
#     n, m = size(x)
#     r_idx = 1:n
#     fact && factorize!(s, A;
#         update=update)
#
#     x .= b
#     for j = 1:m
#         xv = @views x[r_idx, j]
#         solve!(solver.F, xv)
#     end
# end
#
#
#
# struct QRSolver113{T}
#     factorization::QR{T,<:AbstractMatrix{T}}
#     Q::LinearAlgebra.QRPackedQ{T, <:AbstractMatrix{T}}
# end
#
# function QRSolver113(A::AbstractMatrix)
#     factorization = LinearAlgebra.qrfactUnblocked!(A)
#     QRSolver113(factorization, deepcopy(factorization.Q))
# end
#
# function factorize!(solver::QRSolver113{T}, A::AbstractMatrix{T}) where T
#     factorize!(solver.factorization, A)
#     Q .= solver.factorization.Q
# end
#
# function factorize!(factorization::QR{T}, B::AbstractMatrix{T}) where T
#     # https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/qr.jl
#     A = factorization.factors
#     τ = factorization.τ
#
#     #reset
#     A .= B # reset the factors to the value of the new matrix we want to factorize
#     τ .= 0.0
#
#     LinearAlgebra.require_one_based_indexing(A)
#     m, n = size(A)
#     # τ = zeros(T, min(m,n))
#     for k = 1:min(m - 1 + !(T<:Real), n)
#         x = view(A, k:m, k)
#         τk = LinearAlgebra.reflector!(x)
#         τ[k] = τk
#         LinearAlgebra.reflectorApply!(x, τk, view(A, k:m, k + 1:n))
#     end
#     # LinearAlgebra.QR(A, τ)
#     return nothing
# end
#
# import Base.(\)
# function (\)(solver::QRSolver113{T}, B::VecOrMat{T}) where T
#     return solver.factorization \ B
# end
#
#
#
# n = 30
# A1 = rand(n,n)
# A2 = rand(n,n)
# b = rand(n)
# x1 = A1 \ b
# x2 = A2 \ b
#
# s = QRSolver113(A1)
# xf1 = s \ b
# norm(xf1 - x1, Inf)
# @benchmark s \ b
# @benchmark factorization \ b
#
#
# update!(factorization, A2)
# xf2 = factorization \ b
# norm(xf2 - x2, Inf)
#
#
#
# function linear_solve!(solver::QRSolver113{T}, x::Vector{T}, A::AbstractMatrix{T}, b::Vector{T};
#     fact=true,
#     update=true) where {Tv<:AbstractFloat,Ti<:Integer}
#
#     fact && factorize!(solver, A;
#         update=update) # factorize
#
#     ldiv!(x, s.factorization, b)
#     solve!(solver.F, x) # solve
# end
#
# x = zeros(30)
# b = rand(30)
# factorization
# ldiv!(x, factorization, b)
# x
#
# function ttt(x, factorization, b)
#     ldiv!(x, factorization, b)
#     return nothing
# end
#
#
#
# function my_ldiv!(Y::AbstractVector, A::Factorization, B::AbstractVector)
#     LinearAlgebra.require_one_based_indexing(Y, B)
#     m, n = size(A, 1), size(A, 2)
#     if m > n
#         Bc = copy(B)
#         ldiv!(A, Bc)
#         return copyto!(Y, 1, Bc, 1, n)
#     else
#         return ldiv!(A, copyto!(Y, B))
#     end
# end
#
# ldiv!(factorization, x)
# Main.@code_warntype ldiv!(factorization, x)
#
# @benchmark $ldiv!($factorization, $x)
#
#
#
# @benchmark $my_ldiv!($x, $factorization, $b)
# # @benchmark $ttt($x, $factorization, $b)
#
# @benchmark (\)($factorization, $b)
# @benchmark $ldiv!($factorization, $b)
# ldiv!(Fadj::Adjoint{<:Any,<:Union{QR,QRCompactWY,QRPivoted}}, B::AbstractVecOrMat)
#
#
#
# n = 100
# A = rand(n,n)
# b = rand(n)
# x = zeros(n)
#
# @benchmark Alu = lu(A)
# @benchmark $ldiv!($x, $Alu, $b)
#
# LinearAlgebra.lu!(Alu, A)
#
# Alu.L
# Alu.U)
#
#
# function myldiv!(x::Vector{T}, solver::QRSolver113{T}, b::Vector{T}) where T
#     # m, n = size(A)
#     # m < n && return _wide_qr_ldiv!(A, b)
#
#     # triu!(getfield(A, :factors)[1:min(m,n), 1:n])
#     F = solver.factorization
#     solver.Q .= LinearAlgebra.QRPackedQ(getfield(F, :factors), F.τ)
#
#     # lmul!(adjoint(A.Q), view(b, 1:m, :))
#     # lmul!(adjoint(A.Q), b)
#     mul!(x, Q, b)
#     # A.Q
#     # R = A.factors
#     # ldiv!(x, UpperTriangular(view(R,1:n,:)), view(b, 1:n, :))
#     return nothing
# end
#
# A = rand(30,30)
# x = zeros(30)
# b = rand(30)
# solver = QRSolver113(A)
# factorization = LinearAlgebra.qrfactUnblocked!(A)
# # myldiv!(x, factorization, b)
# QQ = factorization.Q
# QQ.factors
# factorization.factors
#
# function lmul!(A::QRPackedQ, B::AbstractVecOrMat)
#     require_one_based_indexing(B)
#     mA, nA = size(A.factors)
#     mB, nB = size(B,1), size(B,2)
#     if mA != mB
#         throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
#     end
#     Afactors = A.factors
#     @inbounds begin
#         for k = min(mA,nA):-1:1
#             for j = 1:nB
#                 vBj = B[k,j]
#                 for i = k+1:mB
#                     vBj += conj(Afactors[i,k])*B[i,j]
#                 end
#                 vBj = A.τ[k]*vBj
#                 B[k,j] -= vBj
#                 for i = k+1:mB
#                     B[i,j] -= Afactors[i,k]*vBj
#                 end
#             end
#         end
#     end
#     B
# end
#
#
# Main.@code_warntype myldiv!(x, factorization, b)
# @benchmark $myldiv!($x, $factorization, $b)
#
#
# myldiv!()
#
# fieldnames(typeof(factorization))
#
# factorization.Q.factors
#
# factorization.Val(:R)
#
# factorization
