using PATHSolver
using JuMP
using BenchmarkTools

M = [
     0  0 -1 -1
     0  0  1 -2
     1 -1  2 -2
     1  2 -2  4
     ]
q = [2, 2, -2, -6]

model = Model(PATHSolver.Optimizer)

set_optimizer_attribute(model, "output", "no")

@variable(model, γ[1:4] >= 0)

@constraint(model, M * γ .+ q ⟂ γ)

optimize!(model)
@benchmark $optimize!($model)

value.(γ)
M * value.(γ) + q

termination_status(model)



function residual(primals, duals, slacks)
    M = [
         0  0 -1 -1
         0  0  1 -2
         1 -1  2 -2
         1  2 -2  4
         ]
    q = [2, 2, -2, -6]

    res = slacks - (M * duals + q)
    return res
end

num_primals = 0
num_cone = 4
solver = Solver(residual, num_primals, num_cone,
    options=Options(
        verbose=false,
        sparse_solver=false,
        # differentiate=true,
        compressed_search_direction=true,
        residual_tolerance=1e-10,
        complementarity_tolerance=1e-10,
    ))

solve!(solver)
Main.@code_warntype solve!(solver)
Main.@profiler [solve!(solver) for i=1:100000]
@benchmark $solve!($solver)



M = convert(
     SparseArrays.SparseMatrixCSC{Cdouble,Cint},
     SparseArrays.sparse([
         0 0 -1 -1
         0 0 1 -2
         1 -1 2 -2
         1 2 -2 4
     ]),
)

status, z, info = PATHSolver.solve_mcp(
    M,
    Float64[2, 2, -2, -6],
    fill(0.0, 4),
    fill(10.0, 4),
    [0.0, 0.0, 0.0, 0.0];
    output = "no",
)

arg0 = Float64[2, 2, -2, -6]
arg1 = fill(0.0, 4)
arg2 = fill(10.0, 4)
arg3 = [0.0, 0.0, 0.0, 0.0]
@benchmark $(PATHSolver.solve_mcp)(
    $M,
    $arg0,
    $arg1,
    $arg2,
    $arg3;
    output = "no",
)


@test status == PATHSolver.MCP_Solved
@test isapprox(z, [2.8, 0.0, 0.8, 1.2])

z .* (M*z + q)
z
M*z + q
