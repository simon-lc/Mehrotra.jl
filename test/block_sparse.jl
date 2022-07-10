@testset "BlockSparse" begin
    n = 15
    A1 = sprand(1n,1n,0.1)
    A2 = sprand(1n,1n,0.1)
    A3 = sprand(1n,1n,0.1)
    A4 = sprand(1n,1n,0.1)
    A5 = sprand(1n,1n,0.1)
    A6 = sprand(1n,1n,0.1)
    R1 = (Vector(0n+1:1n), Vector(0n+1:1n))
    R2 = (Vector(0n+1:1n), Vector(1n+1:2n))
    R3 = (Vector(0n+1:1n), Vector(2n+1:3n))
    R4 = (Vector(1n+1:2n), Vector(0n+1:1n))
    R5 = (Vector(1n+1:2n), Vector(1n+1:2n))
    R6 = (Vector(1n+1:2n), Vector(2n+1:3n))
    block_matrix0 = Mehrotra.BlockSparse116(2n, 3n, [A1,A2,A3,A4,A5,A6], [R1,R2,R3,R4,R5,R6])
    block_matrix1 = Mehrotra.BlockSparse116(2n, 3n, [A1,A2,A3,A4,A5,A6], [R1,R2,R3,R4,R5,R6])

    block_matrix0.matrix .= 0.0
    Mehrotra.fill!(block_matrix0, A1, :b1)
    Mehrotra.fill!(block_matrix0, A2, :b2)
    Mehrotra.fill!(block_matrix0, A3, :b3)
    Mehrotra.fill!(block_matrix0, A4, :b4)
    Mehrotra.fill!(block_matrix0, A5, :b5)
    Mehrotra.fill!(block_matrix0, A6, :b6)
    @test norm(block_matrix0.matrix - block_matrix1.matrix, Inf) < 1e-10

    block_matrix2 = Mehrotra.BlockSparse116(2n, 3n, 2 .* [A1,A2,A3,A4,A5,A6], [R1,R2,R3,R4,R5,R6])
    Mehrotra.fill!(block_matrix0, 2A1, :b1)
    Mehrotra.fill!(block_matrix0, 2A2, :b2)
    Mehrotra.fill!(block_matrix0, 2A3, :b3)
    Mehrotra.fill!(block_matrix0, 2A4, :b4)
    Mehrotra.fill!(block_matrix0, 2A5, :b5)
    Mehrotra.fill!(block_matrix0, 2A6, :b6)
    @test norm(block_matrix0.matrix - block_matrix2.matrix, Inf) < 1e-10

    num_allocs = @ballocated $(Mehrotra.fill!)($block_matrix0, $A6, $:b6)
    @test num_allocs == 0
end
