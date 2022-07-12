@testset "linear solver: dense LU" begin
    ############################################################################
    # vector
    ############################################################################
    n = 10
    A = rand(n,n)
    b = rand(n)
    x = zeros(n)
    solver = Mehrotra.lu_solver(A)

    Mehrotra.linear_solve!(solver, x, 2A, b)
    @test norm(2A*x - b, Inf) < 1e-12

    # Main.@code_warntype linear_solve!(solver, x, A, b)
    @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    @test num_allocs == 0

    ############################################################################
    # matrix
    ############################################################################
    n = 10
    A = rand(n,n)
    b = rand(n,2)
    x = zeros(n,2)
    solver = Mehrotra.lu_solver(A)

    Mehrotra.linear_solve!(solver, x, 2A, b)
    @test norm(2A*x - b, Inf) < 1e-12

    # Main.@code_warntype linear_solve!(solver, x, A, b)
    @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    @test num_allocs == 0
end

@testset "linear solver: sparse LU" begin
    ############################################################################
    # vector
    ############################################################################
    n = 10
    A = sprand(n, n, 0.9)
    b = rand(n)
    x = zeros(n)
    solver = Mehrotra.sparse_lu_solver(A)

    Mehrotra.linear_solve!(solver, x, 2A, b)
    @test norm(2A*x - b, Inf) < 1e-12

    # Main.@code_warntype linear_solve!(solver, x, A, b)
    # @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    # num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    # @test num_allocs == 0

    ############################################################################
    # matrix
    ############################################################################
    n = 10
    A = sprand(n, n, 0.9)
    b = rand(n,2)
    x = zeros(n,2)
    solver = Mehrotra.sparse_lu_solver(A)

    Mehrotra.linear_solve!(solver, x, 2A, b)
    @test norm((2A)*x - b, Inf) < 1e-12

    # Main.@code_warntype linear_solve!(solver, x, A, b)
    # @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    # num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    # @test num_allocs == 0
end

@testset "linear solver: sparse QDLDL" begin
    ############################################################################
    # vector
    ############################################################################
    n = 10
    Ap = sprand(n, n, 0.9)
    Ap = Ap * Ap'
    Am = sprand(n, n, 0.9)
    Am = Am * Am'
    A = cat(Ap, Am, dims=(1,2))
    b = rand(2n)
    x = zeros(2n)
    solver = Mehrotra.ldl_solver(A)

    Mehrotra.linear_solve!(solver, x, 2A, b)
    @test norm(2A * x - b, Inf) < 1e-12

    # Main.@code_warntype Mehrotra.linear_solve!(solver, x, A, b)
    # @benchmark $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
    @ballocated $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
    num_allocs = @ballocated $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
    @test num_allocs <= 32

    ############################################################################
    # matrix
    ############################################################################
    n = 10
    Ap = sprand(n, n, 0.9)
    Ap = Ap * Ap'
    Am = sprand(n, n, 0.9)
    Am = Am * Am'
    A = cat(Ap, Am, dims=(1,2))
    b = rand(2n,2)
    x = zeros(2n,2)
    solver = Mehrotra.ldl_solver(A)

    Mehrotra.linear_solve!(solver, x, 2A, b)
    @test norm(2A * x - b, Inf) < 1e-12

    # Main.@code_warntype Mehrotra.linear_solve!(solver, x, A, b)
    # @benchmark $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
    @ballocated $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
    num_allocs = @ballocated $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
    @test num_allocs <= 672
end








# solver.A_sparse
# # factorize!(solver, A)
# # x .= b
# # solve!(solver.F, x)
# # norm(A*x - b, Inf)
# # At = triu(A)
# # @benchmark $At .= $triu($A)


# update_values!(solver.F, 1:length(solver.A_sparse.nzval), solver.A_sparse.nzval)
# F = solver.F
# A_sparse = solver.A_sparse
# update_values!(F, 1:110, A_sparse.nzval)
# A_nzval = A_sparse.nzval
# @benchmark $update_values!($F, 1:110, $A_nzval)


# solver.A_sparse
# factorize!(solver, 2A)
# Mehrotra.linear_solve!(solver, x, 2A, b)
# @test norm(2A*x - b, Inf) < 1e-10
# A*x - b

# W = F.workspace
# QDLDL.factor!(solver.F.workspace, false)
# Main.@code_warntype refactor!(solver.F)
# Main.@code_warntype QDLDL.factor!(solver.F.workspace, false)
# @benchmark $refactor!($F)
# @benchmark $(QDLDL.factor!)($W, false)


# function refactor!(F::QDLDLFactorisation)

#     #It never makes sense to call refactor for a logical
#     #factorization since it will always be the same.  Calling
#     #this function implies that we want a numerical factorization

#     F.logical[] = false  #in case not already

#     factor!(F.workspace,F.logical[])
# end

# fieldnames(typeof(A))
# A.colptr
# A.rowval
# # @benchmark $triu!($A)

# Main.@code_warntype linear_solve!(solver, x, A, b)
# @benchmark $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
# @ballocated $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
# num_allocs = @ballocated $(Mehrotra.linear_solve!)($solver, $x, $A, $b)
# @test num_allocs <= 32

# ############################################################################
# # matrix
# ############################################################################
# n = 10
# A = sprand(n, n, 0.9)
# b = rand(n,2)
# x = zeros(n,2)
# solver = Mehrotra.sparse_lu_solver(A)

# Mehrotra.linear_solve!(solver, x, A, b)
# @test norm(A*x - b, Inf) < 1e-12

# # Main.@code_warntype linear_solve!(solver, x, A, b)
# # @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
# # num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
# # @test num_allocs == 0
# # end
