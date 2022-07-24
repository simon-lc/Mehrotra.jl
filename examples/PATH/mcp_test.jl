using PATHSolver
using JuMP
using BenchmarkTools

M = [
    0 0 -1 -1
    0 0 1 -2
    1 -1 2 -2
    1 2 -2 4
]
q = [2; 2; -2; -6]
function F(n::Cint, x::Vector{Cdouble}, f::Vector{Cdouble})
    @assert n == length(x) == length(f) == 4
    f[1] = -x[3]^2 - x[4] + q[1]
    f[2] = x[3]^3 - 2x[4]^2 + q[2]
    f[3] = x[1]^5 - x[2] + 2x[3] - 2x[4] + q[3]
    f[4] = x[1] + 2x[2]^3 - 2x[3] + 4x[4] + q[4]
    return Cint(0)
end
function J(
    n::Cint,
    nnz::Cint,
    x::Vector{Cdouble},
    col::Vector{Cint},
    len::Vector{Cint},
    row::Vector{Cint},
    data::Vector{Cdouble},
    )

    JAC = [
        0 0 -2x[3] -1
        0 0 3x[3]^2 -4x[4]
        5x[1]^4 -1 2 -2
        1 6x[2] -2 4
    ]
    @assert n == length(x) == length(col) == length(len) == 4
    @assert nnz == length(row) == length(data)
    i = 1
    for c in 1:n
        col[c] = i
        len[c] = 0
        for r in 1:n
            if !iszero(JAC[r, c])
                data[i] = JAC[r, c]
                row[i] = r
                len[c] += 1
                i += 1
            end
        end
    end
    return Cint(0)
end

status, z, info = PATHSolver.solve_mcp(
    F,
    J,
    fill(0.0, 4),
    fill(10.0, 4),
    [1.0, 1.0, 1.0, 1.0];
    output = "yes",
)
@benchmark $(PATHSolver.solve_mcp)(
    F,
    J,
    $(fill(0.0, 4)),
    $(fill(10.0, 4)),
    $([1.0, 1.0, 1.0, 1.0]);
    output = "no",
)

@test status == PATHSolver.MCP_Solved
@test isapprox(z, [1.28475, 0.972916, 0.909376, 1.17304], atol = 1e-4)

z .* (M*z + q)





# MCP conversion
    # Complementarity constraints
    # https://jump.dev/JuMP.jl/stable/manual/constraints/#Complementarity-constraints
    # A mixed complementarity constraint F(x) ⟂ x consists of finding x in the interval [lb, ub], such that the following holds:
    # F(x) == 0 if lb < x < ub
    # F(x) >= 0 if lb == x
    # F(x) <= 0 if x == ub

    # |A B 0| |Δy|   |-optimality    |
    # |C 0 I|×|Δz| = |-slack_equality|
    # |0 S Z| |Δs|   |-cone_product  |
    # For the equivalence, we choose x = [y, z] f(x) = [Ay + Bz + e; Cy + s + f]
    # lb = [-Inf, 0]
    # ub = [+Inf, +Inf]

Random.seed!(0)
num_primals = 20
num_cone = 20
As = Diagonal(rand(num_primals)) #ones(num_primals, num_primals)
A = As * As'
B = [I(num_cone); zeros(num_primals-num_cone, num_cone)]
C = -B'
e = rand(num_primals)
f = -ones(num_cone)

# PATH solver
lb = [-Inf*ones(num_primals); 0*ones(num_cone)]
ub = [+Inf*ones(num_primals); +Inf*ones(num_cone)]

model = Model(PATHSolver.Optimizer)
set_optimizer_attribute(model, "output", "yes")
@variable(model, lb[i] <= x[i = 1:num_primals+num_cone] <= ub[i])

function path_equality(x)
    y = x[1:num_primals]
    z = x[num_primals .+ (1:num_cone)]
    # return [A*y + B*z + e; -C*y - f]
    return [A*y + B*z + e; C*y + f]
end

@constraint(model, path_equality(x) ⟂ x)

optimize!(model)
termination_status(model)
@benchmark $optimize!($model)

yz0 = value.(x)
path_equality(value.(x))
y0 = yz0[1:num_primals]
z0 = yz0[num_primals .+ (1:num_cone)]
s0 = C * y0 + f
equality(y0, z0, s0)
cone_prod = z0 .* s0


# MEHROTRA solver
function equality(primals, duals, slacks)
    y = primals
    z = duals
    s = slacks
    res = [
        A*y + B*z + e;
        s - (C * y + f);
        ]
    return res
end

solver = Solver(equality, num_primals, num_cone,
    options=Options(
        verbose=false,
        sparse_solver=true,
        compressed_search_direction=true,
        residual_tolerance=1e-10,
        complementarity_tolerance=1e-10,
    ))

solve!(solver)
# set_variables!(solver, [y0; z0; s0])
# solver.options.warm_start = false
solve!(solver)
@benchmark $solve!($solver)
