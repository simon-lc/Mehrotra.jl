using Mehrotra
using MeshCat

include("particle_utils.jl")

vis = Visualizer()
render(vis)

function residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    p2, v15, u, timestep, mass, gravity, friction_coefficient, side = unpack_parameters(parameters)

    v25 = y
    p1 = p2 - timestep * v15
    p3 = p2 + timestep * v25

    γ = z[1:1]
    β = z[2:4]

    sγ = s[1:1]
    sβ = s[2:4]

    N = [0 0 1]
    D = [1 0 0;
         0 1 0]

    vtan = D * v25

    res = [
        mass * (p3 - 2p2 + p1)/timestep - timestep * mass * [0,0, gravity] - N' * γ - D' * β[2:3] - u * timestep;
        sγ - (p3[3:3] .- side/2);
        sβ[2:3] - vtan;
        β[1:1] - friction_coefficient * γ;
        # z ∘ s .- κ[1];
        ]
    return res
end

num_primals = 3
num_cone = 4
num_parameters = 14
idx_nn = collect(1:1)
idx_soc = [collect(2:4)]
p2 = [1,1,1.0]
v15 = [0,-1,1.0]
u = [0.4, 0.8, 0.9]
side = 0.5
parameters = [p2; v15; u; 0.01; 1.0; -9.81; 0.0; side]


primals = zeros(num_primals)
duals = ones(num_cone)
slacks = ones(num_cone)
residual(primals, duals, slacks, parameters)

solver = Solver(residual, num_primals, num_cone,
    parameters=parameters,
    nonnegative_indices=idx_nn,
    second_order_indices=idx_soc,
    options=Options(
        max_iterations=30,
        verbose=true,
        complementarity_tolerance=1e-4,
        residual_tolerance=1e-4,
        differentiate=true,
        )
    )

solve!(solver)
solver.data.solution_sensitivity

solver.options.residual_tolerance
solver.options.complementarity_tolerance
solver.solution.duals
solver.solution.duals
solver.solution.duals
solver.solution.duals




















function active_set_solve!(solver, κ)
    # function active_residual(solver, y, z)
    #     # solver = deepcopy(solver0)
    #     e = [1,1,0,0]
    #     Z = Matrix(Diagonal([z[1]; z[2] * ones(3)]))
    #     Z[2,2:4] .= z[2:4]
    #     Z[2:4,2] .= z[2:4]
    #     s = κ * Z \ e
    #
    #     # solver.solution.primals .= y
    #     # solver.solution.duals .= z
    #     # solver.solution.slacks .= s
    #     #
    #     # residual!(solver.data, solver.problem, solver.indices, solver.solution, κ)
    #     # # return Vector(deepcopy(solver.data.residual.equality))
    #     # # return solver.data.residual.equality
    #     # return deepcopy(solver.data.residual.equality)
    #     return residual(y, z, s, parameters)
    # end

    ny = solver.dimensions.primals
    nz = solver.dimensions.duals
    for i = 1:10
        @show i
        yzs = solver.solution.all
        JF = FiniteDiff.finite_difference_jacobian(yzs ->
            residual(yzs[1:ny], yzs[ny .+ (1:nz)], yzs[ny + nz .+ (1:nz)], parameters), yzs)
        J = JF[1:ny+nz, 1:ny+nz]

        # e = [1,1,0,0.0]
        # dsdz = κ * FiniteDiff.finite_difference_jacobian(z -> κ * Z_matrix(z) \ e, yzs[ny .+ (1:nz)])
        dsdz = dsdz_fct(solver.solution.duals, κ)
        drsds = JF[nx .+ (1:nz), nx + nz .+ (1:nz)]
        J[nx .+ (1:nz), nx .+ (1:nz)] += drsds * dsdz

        r = residual(yzs[1:ny], yzs[ny .+ (1:nz)], yzs[ny + nz .+ (1:nz)], parameters)
        Δ = - J \ r
        @show J
        @show norm(r)
        @show norm(Δ)
        Δy = Δ[1:ny]
        Δz = Δ[ny .+ (1:nz)]

        Δs = dsdz * Δz

        residual!(solver.data, solver.problem, solver.indices, solver.solution, κ)
        equality_violation = norm(solver.data.residual.equality, Inf)
        cone_product_violation = norm(solver.data.residual.cone_product, Inf)

        println("equality = ", scn(equality_violation))
        println("cone prd = ", scn(cone_product_violation))

        α = 1.0
        # cone search duals
        α = cone_search(α, solver.solution.duals, Δz,
            solver.indices.cone_nonnegative, solver.indices.cone_second_order;
            τ_nn=0.99, τ_soc=0.99, ϵ=1e-14)
        # cone search slacks
        α = cone_search(α, solver.solution.slacks, Δs,
            solver.indices.cone_nonnegative, solver.indices.cone_second_order;
            τ_nn=0.99, τ_soc=0.99, ϵ=1e-14)

        for j = 1:20
            solver.candidate.primals .= solver.solution.primals + α * Δy
            solver.candidate.duals .= solver.solution.duals + α * Δz
            solver.candidate.slacks .= solver.solution.slacks + α * Δs


            residual!(solver.data, solver.problem, solver.indices, solver.candidate, κ)
            equality_violation_candidate = norm(solver.data.residual.equality, Inf)
            cone_product_violation_candidate = norm(solver.data.residual.cone_product, Inf)
            # Test progress
            if (equality_violation_candidate <= equality_violation ||
                cone_product_violation_candidate <= cone_product_violation)
                break
            end

            # decrease step size
            α = 0.5 * α
        end

        solver.solution.primals .= solver.solution.primals + α * Δy
        solver.solution.duals .= solver.solution.duals + α * Δz
        solver.solution.slacks .= solver.solution.slacks + α * Δs

        # dredy =
        # dredz =
        # dreds =
        # J = [
        #     dredy dredz + dreds * dsdz;
        #     drsdy drsdz + drsds * dsdz;
        # ]
    end


    return nothing
end

z = [1e-3, 1e-4, 1e-5, 1e-6]
e = [1,1,0,0]
Z = Matrix(Diagonal([z[1]; z[2] *ones(3)]))
Z[2,2:4] .= z[2:4]
Z[2:4,2] .= z[2:4]
Z

active_set_solve!(solver, 1e-4)
solver.solution.duals
solver.solution.slacks


function Z_matrix(z)
    Z = Matrix(Diagonal([z[1]; z[2] *ones(3)]))
    Z[2,2:4] .= z[2:4]
    Z[2:4,2] .= z[2:4]
    return Z
end




@variables zv[1:4]
@variables κv
zv = Symbolics.scalarize(zv)
Zv = Z_matrix(zv)
out = κv * Zv \ e
jac = Symbolics.jacobian(out, zv)
dsdz_fct = eval(build_function(jac, zv, κv)[1])
dsdz_fct(rand(4), 1e-4)

dsdz = κ * FiniteDiff.finite_difference_jacobian(z -> κ * Z_matrix(z) \ e, yzs[ny .+ (1:nz)])


residual!(solver.data, solver.problem, solver.indices, solver.solution, [solver.options.complementarity_tolerance])
residual!(solver.data, solver.problem, solver.indices, solver.solution, [solver.options.complementarity_tolerance])
residual!(solver.data, solver.problem, solver.indices, solver.solution, [0.0])

solver.data.residual.cone_product
solver.solution

variables = solver.solution.all
r, ∇r = residual_complex(solver, variables)

FiniteDiff.finite_difference_jacobian(variables -> residual(
    variables[1:num_primals],
    variables[num_primals .+ (1:num_cone)],
    variables[num_primals + num_cone .+ (1:num_cone)],
    parameters,
    ), variables)



H = 300
p2 = [1,1,1.0]
v15 = [0,-4,1.0]
u = [0(rand(3) .- 0.5) for i=1:H]
p, v = simulate_particle(solver, p2, v15, u, friction_coefficient=0.3)

plot(hcat(p...)')

setobject!(vis[:particle], HyperRectangle(Vec(-side/2, -side/2, -side/2), Vec(side, side, side)))

for i = 1:H
    settransform!(vis[:particle], MeshCat.Translation(p[i]...))
    sleep(0.01)
end

n = 3
idx_nn = Vector{Int}()
idx_soc = [1:n]

x = rand(n)
y = rand(n)
Δx = 0.05 * (rand(n) .- 0.5)
Δy = 0.05 * (rand(n) .- 0.5)

α = 0.1
c0 = cone_product(x, y, idx_nn, idx_soc)
cx = cone_product(Δx, y, idx_nn, idx_soc)
cy = cone_product(x, Δy, idx_nn, idx_soc)
cxy = cone_product(Δx, Δy, idx_nn, idx_soc)
c2 = cone_product(x+Δx, y+Δy, idx_nn, idx_soc)
c2 - c0 - cx - cy - cxy

function max_alpha(z, s, Δz, Δs, idx_nn, idx_soc, options; α=1.0)
    tolerance = max(options.complementarity_tolerance - options.residual_tolerance, 0.0)
    for i = 1:10000
        c = cone_product(z+α*Δz, s+α*Δs, idx_nn, idx_soc)
        all(c .>= tolerance) && break
        α *= 0.98
    end
    return α
end
0.98^1000
