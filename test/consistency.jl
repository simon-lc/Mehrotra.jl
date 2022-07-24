include(joinpath(Mehrotra.module_dir(), "examples/benchmark_problems/lcp_utils.jl"))

@testset "Consistency" begin
    # solve without differentiating
    options = Mehrotra.Options(
        verbose=false,
        differentiate=false,
        )

    solver = random_lcp(; num_primals=6, num_cone=12,
        cone_type=:non_negative_cone,
        options=options,
        seed=1,
        )

    @test solver.consistency.solved[1] == false
    @test solver.consistency.differentiated[:all] == false
    Mehrotra.solve!(solver)
    @test solver.consistency.solved[1] == true
    @test solver.consistency.differentiated[:all] == false

    # solve and differentiate
    solver.options.differentiate = true
    Mehrotra.solve!(solver)
    @test solver.consistency.solved[1] == true
    @test solver.consistency.differentiated[:all] == true

    # initializing primals breaks solved, but maintains differentiated
    Mehrotra.initialize_primals!(solver)
    @test solver.consistency.solved[1] == false
    @test solver.consistency.differentiated[:all] == true
    Mehrotra.solve!(solver)
    @test solver.consistency.solved[1] == true
    @test solver.consistency.differentiated[:all] == true

    # initializing duals breaks solved, but maintains differentiated
    Mehrotra.initialize_duals!(solver)
    @test solver.consistency.solved[1] == false
    @test solver.consistency.differentiated[:all] == true
    Mehrotra.solve!(solver)
    @test solver.consistency.solved[1] == true
    @test solver.consistency.differentiated[:all] == true

    # initializing slacks breaks solved, but maintains differentiated
    Mehrotra.initialize_slacks!(solver)
    @test solver.consistency.solved[1] == false
    @test solver.consistency.differentiated[:all] == true
    Mehrotra.solve!(solver)
    @test solver.consistency.solved[1] == true
    @test solver.consistency.differentiated[:all] == true

    # setting variables breaks solved, but maintains differentiated
    variables = rand(solver.dimensions.variables)
    Mehrotra.set_variables!(solver, variables)
    @test solver.consistency.solved[1] == false
    @test solver.consistency.differentiated[:all] == true
    Mehrotra.solve!(solver)
    @test solver.consistency.solved[1] == true
    @test solver.consistency.differentiated[:all] == true

    # setting the parameters breaks solved and differentiated
    Random.seed!(0)
    parameters = rand(solver.dimensions.parameters)
    Mehrotra.set_parameters!(solver, parameters)
    @test solver.consistency.solved[1] == false
    @test solver.consistency.differentiated[:all] == false
    Mehrotra.solve!(solver)
    @test solver.consistency.solved[1] == true
    @test solver.consistency.differentiated[:all] == true
end
