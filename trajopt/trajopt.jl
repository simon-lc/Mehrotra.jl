using LinearAlgebra
using Plots
using DirectTrajectoryOptimization

const DTO = DirectTrajectoryOptimization

# ## horizon
T = 101

# ## cartpole
num_state = 4
num_action = 1
num_parameter = 0

function cartpole(x, u, w)
    mc = 1.0
    mp = 0.2
    l = 0.5
    g = 9.81

    q = x[1:2]
    qd = x[3:4]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    Hinv = 1.0 / (H[1, 1] * H[2, 2] - H[1, 2] * H[2, 1]) * [H[2, 2] -H[1, 2]; -H[2, 1] H[1, 1]]

    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp*g*l*s]
    B = [1, 0]

    qdd = -Hinv * (C*qd + G - B*u[1])


    return [qd; qdd]
end

function rk3_explicit(x, u, w)
    h = 0.05 # timestep

    k1 = h * cartpole(x, u, w)
    k2 = h * cartpole(x + 0.5 * k1, u, w)
    k3 = h * cartpole(x - k1 + 2.0 * k2, u, w)

    return x + (k1 + 4.0 * k2 + k3) / 6.0
end

function rk3_implicit(y, x, u, w)
    return y - rk3_explicit(x, u, w)
end

# ## model
dt = Dynamics(rk3_implicit, num_state, num_state, num_action,
    num_parameter=num_parameter)
dyn = [dt for t = 1:T-1]

# ## initialization
x1 = [0.0; 0.0; 0.0; 0.0]
xT = [0.0; π; 0.0; 0.0]

# ## objective
Q = 1.0e-2
R = 1.0e-1
Qf = 1.0e2

ot = (x, u, w) -> 0.5 * Q * dot(x - xT, x - xT) + 0.5 * R * dot(u, u)
oT = (x, u, w) -> 0.5 * Qf * dot(x - xT, x - xT)
ct = Cost(ot, num_state, num_action,
    num_parameter=num_parameter)
cT = Cost(oT, num_state, 0,
    num_parameter=num_parameter)
obj = [[ct for t = 1:T-1]..., cT]

# ## constraints
u_bnd = 3.0
bnd1 = Bound(num_state, num_action,
    action_lower=[-u_bnd],
    action_upper=[u_bnd])
bndt = Bound(num_state, num_action,
    action_lower=[-u_bnd],
    action_upper=[u_bnd])
bndT = Bound(num_state, 0)
bounds = [bnd1, [bndt for t = 2:T-1]..., bndT]

cons = [
            Constraint((x, u, w) -> x - x1, num_state, num_action),
            [Constraint() for t = 2:T-1]...,
            Constraint((x, u, w) -> x - xT, num_state, 0)
       ]

# ## problem
solver = DTO.Solver(dyn, obj, cons, bounds,
    options=Options{Float64}())

# ## initialize
u_guess = [0.01 * ones(num_action) for t = 1:T-1]
x_rollout = [x1]
for t = 1:T-1
    push!(x_rollout, rk3_explicit(x_rollout[end], u_guess[t], zeros(num_parameter)))
end

initialize_states!(solver, x_rollout)
initialize_controls!(solver, u_guess)

# ## solve
@time DTO.solve!(solver)

# ## solution
x_sol, u_sol = get_trajectory(solver)

@show x_sol[1]
@show x_sol[T]

# ## state
plot(hcat(x_sol...)')

# ## control
plot(hcat(u_sol[1:end-1]..., u_sol[end-1])', linetype = :steppost)
