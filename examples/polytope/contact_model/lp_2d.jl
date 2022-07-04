################################################################################
# residual
################################################################################
function lp_residual(primals, duals, slacks, parameters; na::Int=0, nb::Int=0, d::Int=0)

    xa, qa, xb, qb, Aa, ba, Ab, bb = unpack_lp_parameters(parameters, na=na, nb=nb, d=d)

    y, z, s = primals, duals, slacks
    za = z[1:na]
    zb = z[na .+ (1:nb)]
    sa = s[1:na]
    sb = s[na .+ (1:nb)]

    # pw is expressed in world's frame
    pw = y[1:d] + (xa + xb) ./ 2
    ϕ = y[d .+ (1:1)]
    # pa is expressed in bodya's frame
    pa = x_2d_rotation(qa)' * (pw - xa)
    # pb is expressed in bodyb's frame
    pb = x_2d_rotation(qb)' * (pw - xb)

    res = [
        x_2d_rotation(qa) * Aa' * za + x_2d_rotation(qb) * Ab' * zb;
        1 - sum(za) - sum(zb);
        sa - (- Aa * pa + ba + ϕ .* ones(na));
        sb - (- Ab * pb + bb + ϕ .* ones(nb));
        # sa .* za;
        # sb .* zb;
    ]
    return res
end

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

################################################################################
# parameters
################################################################################
function unpack_lp_parameters(parameters; na=0, nb=0, d=0)
    off = 0
    xa = parameters[off .+ (1:d)]; off += d
    qa = parameters[off .+ (1:1)]; off += 1
    xb = parameters[off .+ (1:d)]; off += d
    qb = parameters[off .+ (1:1)]; off += 1

    Aa = parameters[off .+ (1:na*d)]; off += na*d
    Aa = reshape(Aa, (na,d))
    ba = parameters[off .+ (1:na)]; off += na

    Ab = parameters[off .+ (1:nb*d)]; off += nb*d
    Ab = reshape(Ab, (nb,d))
    bb = parameters[off .+ (1:nb)]; off += nb

    return xa, qa, xb, qb, Aa, ba, Ab, bb
end

function pack_lp_parameters(xa, qa, xb, qb, Aa, ba, Ab, bb)
    return [xa; qa; xb; qb; vec(Aa); ba; vec(Ab); bb]
end

################################################################################
# solver
################################################################################
function lp_contact_solver(Aa, ba, Ab, bb; d::Int=2,
        options::Options228=Options228(compressed_search_direction=false))
    na = length(ba)
    nb = length(bb)

    xa2 = zeros(d)
    xb2 = zeros(d)
    qa2 = zeros(1)
    qb2 = zeros(1)

    parameters = pack_lp_parameters(xa2, qa2, xb2, qb2, Aa, ba, Ab, bb)
    num_primals = d + 1
    num_cone = na + nb
    idx_nn = collect(1:num_cone)
    idx_soc = [collect(1:0)]

    sized_lp_residual(primals, duals, slacks, parameters) =
        lp_residual(primals, duals, slacks, parameters; na=na, nb=nb, d=d)

    solver = Solver(
            sized_lp_residual,
            num_primals,
            num_cone,
            parameters=parameters,
            nonnegative_indices=idx_nn,
            second_order_indices=idx_soc,
            options=options,
            )
    return solver
end

function set_pose_parameters!(solver::Solver228, xa, qa, xb, qb; na::Int=0, nb::Int=0, d::Int=0)
    _, _, _, _, Aa, ba, Ab, bb = unpack_lp_parameters(solver.parameters, na=na, nb=nb, d=d)
    solver.parameters .= pack_lp_parameters(xa, qa, xb, qb, Aa, ba, Ab, bb)
    return nothing
end

################################################################################
# contact bundle parameters
################################################################################
function contact_bundle(xl, parameters, solver::Solver228;
        na::Int=0, nb::Int=0, d::Int=0)
    # v = variables [xa, qa, xb, qb]
    # ϕ = signed distance function
    # N = ∂ϕ∂v jacobian
    # pa = contact point on body a in world coordinates
    # pb = contact point on body b in world coordinates
    # vpa = ∂pa∂v jacobian, derivative of the contact point location not attached to body a
    # vpb = ∂pb∂v jacobian, derivative of the contact point location not attached to body a

    # solver.parameters .= parameters
    # solver.options.differentiate = true
    # solve!(solver)

    # ϕ = solver.solution.primals[d .+ (1:1)]
    # pa = solver.solution.primals[1:d]
    # pb = solver.solution.primals[1:d]
    # N = solver.data.solution_sensitivity[d .+ (1:1), 1:2d+2]
    # vpa = solver.data.solution_sensitivity[1:d, 1:2d+2]
    # vpb = solver.data.solution_sensitivity[1:d, 1:2d+2]
    #
    # return xl .= pack_contact_bundle(ϕ, pa, pb, N, vpa, vpb)
end

function contact_bundle_jacobian(jac, parameters, solver::Solver228;
        na::Int=0, nb::Int=0, d::Int=0)
    solver.parameters .= parameters
    solver.options.differentiate = true
    solve!(solver)

    ϕ = solver.solution.primals[d .+ (1:1)]
    pa = solver.solution.primals[1:d]
    pb = solver.solution.primals[1:d]
    ∂N = solver.data.solution_sensitivity[d .+ (1:1), 1:2d+2]
    ∂pa = solver.data.solution_sensitivity[1:d, 1:2d+2]
    ∂pb = solver.data.solution_sensitivity[1:d, 1:2d+2]

    # ∂xl∂θl = ∂subvariables / ∂subparameters
    ∂xl∂θl = solver.data.solution_sensitivity
    return pack_contact_bundle(ϕ, pa, pb, N, ∂pa, ∂pb)
end




################################################################################
# contact bundle parameters
################################################################################
function unpack_contact_bundle(parameters; d::Int=0)
    off = 0
    ϕ = parameters[off .+ (1:1)]; off += 1
    pa = parameters[off .+ (1:d)]; off += d
    pb = parameters[off .+ (1:d)]; off += d
    N = parameters[off .+ (1:1*(2d+2))]; off += 1*(2d+2)
    N = reshape(N, (1,2d+2))
    vpa = parameters[off .+ (1:d*(2d+2))]; off += d*(2d+2)
    vpa = reshape(vpa, (d,2d+2))
    vpb = parameters[off .+ (1:d*(2d+2))]; off += d*(2d+2)
    vpb = reshape(vpb, (d,2d+2))
    return ϕ, pa, pb, N, vpa, vpb
end

function pack_contact_bundle(ϕ, pa, pb, N, vpa, vpb)
    return [ϕ; pa; pb; vec(N); vec(vpa); vec(vpb)]
end




################################################################################
# demo
################################################################################
vis = Visualizer()
render(vis)
set_floor!(vis)
set_light!(vis)
set_background!(vis)

Aa = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.10ones(4,2)
ba = 0.5*[
    +1,
    +1,
    +1,
     2,
    ]

Ab = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.10ones(4,2)
bb = 0.5*[
     1,
     1,
     1,
     1,
    ]
na = length(ba)
nb = length(bb)
d = 2

build_2d_polyhedron!(vis, Aa, ba, color=RGBA(0.2,0.2,0.2,0.6), name=:polya)
build_2d_polyhedron!(vis, Ab, bb, color=RGBA(0.8,0.8,0.8,0.6), name=:polyb)

xa2 = [0.4,3.0]
xb2 = [0,4.0]
qa2 = [+0.5]
qb2 = [-0.5]

set_2d_polyhedron!(vis, xa2, qa2, name=:polya)
set_2d_polyhedron!(vis, xb2, qb2, name=:polyb)

contact_solver = lp_contact_solver(Aa, ba, Ab, bb; d=2,
    options=Options228(verbose=true, compressed_search_direction=true, differentiate=true))
set_pose_parameters!(contact_solver, xa2, qa2, xb2, qb2, na=na, nb=nb, d=d)

solve!(contact_solver)
# @benchmark $solve!($contact_solver)
# Main.@profiler [solve!(contact_solver) for i=1:1000]


function search_direction!(solver::Solver228; compressed::Bool=false)
    dimensions = solver.dimensions
    linear_solver = solver.linear_solver
    data = solver.data
    residual = data.residual
    step = data.step

    if compressed
        step = compressed_search_direction!(linear_solver, dimensions, data, residual, step)
    else
        step = uncompressed_search_direction!(linear_solver, dimensions, data, residual, step)
    end
    return step
end

contact_solver_c = lp_contact_solver(Aa, ba, Ab, bb; d=2,
    options=Options228(verbose=true, differentiate=true, compressed_search_direction=true))
contact_solver_u = lp_contact_solver(Aa, ba, Ab, bb; d=2,
    options=Options228(verbose=true, differentiate=true, compressed_search_direction=false))


solve!(contact_solver_c)
solve!(contact_solver_u)
S0 = contact_solver_c.data.solution_sensitivity
S1 = contact_solver_u.data.solution_sensitivity

indices = contact_solver_c.indices
norm(S0[indices.equality, :])
norm(S1[indices.equality, :])
norm(S0[indices.equality, :] - S1[indices.equality, :], Inf)


duals = rand(8)
slacks = rand(8)

initialize_primals!(contact_solver_u)
initialize_duals!(contact_solver_u)
initialize_slacks!(contact_solver_u)
contact_solver_u.solution.duals .= duals
contact_solver_u.solution.slacks .= slacks
evaluate!(contact_solver_u.problem,
    contact_solver_u.methods,
    contact_solver_u.cone_methods,
    contact_solver_u.solution,
    contact_solver_u.parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    cone_constraint=true,
    cone_jacobian=true,
    cone_jacobian_inverse=true,
)

residual!(contact_solver_u.data,
    contact_solver_u.problem,
    contact_solver_u.indices,
    contact_solver_u.solution,
    contact_solver_u.parameters,
    contact_solver_u.central_paths.zero_central_path, compressed=false)
differentiate!(contact_solver_u)
ustep = deepcopy(search_direction!(contact_solver_u, compressed=false))


initialize_primals!(contact_solver_c)
initialize_duals!(contact_solver_c)
initialize_slacks!(contact_solver_c)
contact_solver_c.solution.duals .= duals
contact_solver_c.solution.slacks .= slacks
evaluate!(contact_solver_c.problem,
    contact_solver_c.methods,
    contact_solver_c.cone_methods,
    contact_solver_c.solution,
    contact_solver_c.parameters,
    equality_constraint=true,
    equality_jacobian_variables=true,
    cone_constraint=true,
    cone_jacobian=true,
    cone_jacobian_inverse=true,
)

residual!(contact_solver_c.data,
    contact_solver_c.problem,
    contact_solver_c.indices,
    contact_solver_c.solution,
    contact_solver_c.parameters,
    contact_solver_c.central_paths.zero_central_path, compressed=true)
differentiate!(contact_solver_c)
cstep = deepcopy(search_direction!(contact_solver_c, compressed=true))
norm(cstep - ustep, Inf)

S0 = contact_solver_c.data.solution_sensitivity
S1 = contact_solver_u.data.solution_sensitivity

indices = contact_solver_c.indices
norm(S0[indices.equality, :])
norm(S1[indices.equality, :])
norm(S0[indices.equality, :] - S1[indices.equality, :], Inf)
norm(S0[indices.slacks, :] - S1[indices.slacks, :], Inf)

S0[indices.slacks, :]
S1[indices.slacks, :]


plot(Gray.(abs.(S0[indices.slacks, :] - S1[indices.slacks, :])))
S0[indices.slacks, :]



p = contact_solver.solution.primals[1:d]
ϕ = contact_solver.solution.primals[d+1]

setobject!(vis[:contacta],
    HyperSphere(GeometryBasics.Point(0, ((xa2+xb2)/2 .+ p)...), 0.05),
    MeshPhongMaterial(color=RGBA(1,0,0,1.0)))

contact_bundle(contact_solver, xa2, qa2, xb2, qb2; na=na, nb=nb, d=d, differentiate=true)
