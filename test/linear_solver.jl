@testset "linear solver: dense LU" begin
    ############################################################################
    # vector
    ############################################################################
    n = 10
    A = rand(n,n)
    b = rand(n)
    x = zeros(n)
    solver = Mehrotra.lu_solver(A)

    Mehrotra.linear_solve!(solver, x, A, b)
    @test norm(A*x - b, Inf) < 1e-12

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

    Mehrotra.linear_solve!(solver, x, A, b)
    @test norm(A*x - b, Inf) < 1e-12

    # Main.@code_warntype linear_solve!(solver, x, A, b)
    @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
    @test num_allocs == 0
end



############################################################################
# vector
############################################################################
n = 10
A = sprand(n, n, 0.9)
b = rand(n)
x = zeros(n)
solver = Mehrotra.sparse_lu_solver(A)

Mehrotra.linear_solve!(solver, x, A, b)
@test norm(A*x - b, Inf) < 1e-12

# Main.@code_warntype linear_solve!(solver, x, A, b)
@ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
@test num_allocs == 0

############################################################################
# matrix
############################################################################
n = 10
A = sprand(n, n, 0.9)
b = rand(n,2)
x = zeros(n,2)
solver = Mehrotra.sparse_lu_solver(A)

Mehrotra.linear_solve!(solver, x, A, b)
@test norm(A*x - b, Inf) < 1e-12

# Main.@code_warntype linear_solve!(solver, x, A, b)
@ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
num_allocs = @ballocated $Mehrotra.linear_solve!($solver, $x, $A, $b)
@test num_allocs == 0
