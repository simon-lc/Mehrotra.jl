# ## visualizer
vis = Visualizer()
open(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

# ## setup
using Mehrotra
using IterativeLQR
using LinearAlgebra

################################################################################
# Continuation
################################################################################
function reset!(mechanism::Mechanism183; residual_tolerance=1e-4, complementarity_tolerance=1e-3)
    mechanism.options.residual_tolerance = residual_tolerance
    mechanism.options.complementarity_tolerance = complementarity_tolerance
    return nothing
end

function continuation_callback!(solver::IterativeLQR.Solver, mechanism::Mechanism183; ρ=1.5)
    # contact smoothness continuation
    mechanism.options.residual_tolerance = max(1e-6, mechanism.options.residual_tolerance/ρ)
    mechanism.options.complementarity_tolerance = max(1e-4, mechanism.options.complementarity_tolerance/ρ)

    # visualize current policy
    ū = solver.problem.actions
    x̄ = IterativeLQR.rollout(model, x1, ū)
    # visualize(env, x̄)
    # visualize!(vis, mech, storage, build=false)

    println("r_tol", mechanism.options.residual_tolerance *
        "κ_tol", mechanism.options.complementarity_tolerance)
    return nothing
end

################################################################################
# ## system
################################################################################
gravity = -9.81
timestep = 0.02
friction_coefficient = 0.8
mech = get_convex_drop(;
    timestep=0.05,
    gravity=-9.81,
    mass=1.0,
    inertia=0.2 * ones(1,1),
    friction_coefficient=0.1,
    method_type=:symbolic,
    # method_type=:finite_difference,
    options=Options(
        verbose=false,
        complementarity_tolerance=1e-3,
        compressed_search_direction=true,
        max_iterations=30,
        sparse_solver=true,
        differentiate=false,
        warm_start=true,
        complementarity_correction=0.5,
        )
    );

# ## dimensions
n = mech.dimensions.state
m = mech.dimensions.input
nu_infeasible = 0

################################################################################
# ## simulation test
################################################################################

u_hover = [0; 0.5; 0]
function ctrl!(m, i; u=u_hover)
    set_input!(m, u)
end

xp2 = [+0.0,1.5,-0.25]
vp15 = [-0,0,-0.0]
z0 = [xp2; vp15]
H0 = 150
storage = simulate!(mech, z0, H0, controller=ctrl!)
# visualize!(vis, mech, storage, build=true)
visualize!(vis, mech, storage, build=false)



################################################################################
# ## reference trajectory
################################################################################
zref = [[0, 2, 0, 0, 0, 0.0] for i=1:10]

# ## horizon
T = length(zref)


################################################################################
# ## ILQR problem
################################################################################
# ## model
dyn = IterativeLQR.Dynamics(
    (y, x, u, w) -> dynamics(y, mech, x, u, w),
    (dx, x, u, w) -> dynamics_jacobian_state(dx, mech, x, u, w),
    (du, x, u, w) -> dynamics_jacobian_input(du, mech, x, u, w),
    n, n, m)

model = [dyn for t = 1:T-1]

# ## rollout
x1 = deepcopy(xref[1])
ū = [u_hover for t = 1:T-1]

x̄ = IterativeLQR.rollout(model, x1, ū)
DojoEnvironments.visualize(env, x̄)

# ## objective
############################################################################
qt = [0.3; 0.05; 0.05;
    5e-2 * ones(3);
    1e-3 * ones(3);
    1e-3 * ones(3);
    fill([2, 1e-3], 12)...]
ots = [(x, u, w) -> transpose(x - xref[t]) * Diagonal(timestep * qt) * (x - xref[t]) +
# transpose(u - u_hover) * Diagonal(timestep * 0.01 * ones(m)) * (u - u_hover) for t = 1:T-1]
    transpose(u) * Diagonal(timestep * 0.01 * ones(m)) * u for t = 1:T-1]
oT = (x, u, w) -> transpose(x - xref[end]) * Diagonal(timestep * qt) * (x - xref[end])

cts = [IterativeLQR.Cost(ot, n, m) for ot in ots]
cT = IterativeLQR.Cost(oT, n, 0)
obj = [cts..., cT]


# ## constraints
############################################################################
ul = -1.0 * 1e-3*ones(nu_infeasible)
uu = +1.0 * 1e-3*ones(nu_infeasible)

function contt(x, u, w)
    [
        1e-1 * (ul - u[1:nu_infeasible]);
        1e-1 * (u[1:nu_infeasible] - uu);
    ]
end

function goal(x, u, w)
    Δ = 1e-2 * (x - xref[end])[[1:6;13:2:36]]
    return Δ
end

con_policyt = IterativeLQR.Constraint(contt, n, m, indices_inequality=collect(1:2nu_infeasible))
con_policyT = IterativeLQR.Constraint(goal, n, 0)

cons = [[con_policyt for t = 1:T-1]..., con_policyT]


# ## solver
options = Options(line_search=:armijo,
        max_iterations=50,
        max_dual_updates=12,
        min_step_size=1e-2,
        objective_tolerance=1e-3,
        lagrangian_gradient_tolerance=1e-3,
        constraint_tolerance=1e-4,
        initial_constraint_penalty=1e-1,
        scaling_penalty=10.0,
        max_penalty=1e4,
        verbose=true)

s = IterativeLQR.Solver(model, obj, cons, options=options)

IterativeLQR.initialize_controls!(s, ū)
IterativeLQR.initialize_states!(s, x̄)


# ## solve
local_callback!(solver::IterativeLQR.Solver) = continuation_callback!(solver, env)
reset!(env)
@time IterativeLQR.constrained_ilqr_solve!(s, augmented_lagrangian_callback! = local_callback!)

# ## solution
x_sol, u_sol = IterativeLQR.get_trajectory(s)

# ## visualize
x_view = [[x_sol[1] for t = 1:15]..., x_sol..., [x_sol[end] for t = 1:15]...]
DojoEnvironments.visualize(env, x_view)
