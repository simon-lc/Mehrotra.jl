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
    pw = y[1:d]
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

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

################################################################################
# solver
################################################################################
function lp_contact_solver(Aa, ba, Ab, bb; d::Int=2)
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
            options=Options228(),
            )

    solver.options.compressed_search_direction = false
    solver.options.max_iterations = 30
    # solver.options.verbose = false
    return solver
end

function set_pose_parameters!(solver::Solver228, xa, qa, xb, qb; na::Int=0, nb::Int=0, d::Int=0)
    _, _, _, _, Aa, ba, Ab, bb = unpack_lp_parameters(solver.parameters, na=na, nb=nb, d=d)
    @show size(solver.parameters)
    @show size(pack_lp_parameters(xa, qa, xb, qb, Aa, ba, Ab, bb))
    @show na
    @show nb
    @show d
    solver.parameters .= pack_lp_parameters(xa, qa, xb, qb, Aa, ba, Ab, bb)
    return nothing
end

function contact_bundle(solver::Solver228, xa, qa, xb, qb; na::Int=0, nb::Int=0, d::Int=0,
        differentiate::Bool=true)
    # v = variables [xa, qa, xb, qb]
    # ϕ = signed distance function
    # N = ∂ϕ∂v jacobian
    # pa = contact point on body a in world coordinates
    # pb = contact point on body b in world coordinates
    # ∂pa = ∂pa∂v jacobian, derivative of the contact point location not attached to body a
    # ∂pb = ∂pb∂v jacobian, derivative of the contact point location not attached to body a
    set_pose_parameters!(solver, xa, qa, xb, qb; na=na, nb=nb, d=d)
    solver.options.differentiate = differentiate
    solve!(solver)

    ϕ = solver.solution.primals[d .+ (1:1)]
    pa = solver.solution.primals[1:d]
    pb = solver.solution.primals[1:d]
    N = solver.data.solution_sensitivity[d .+ (1:1), 1:2d+2]
    ∂pa = solver.data.solution_sensitivity[1:d, 1:2d+2]
    ∂pb= solver.data.solution_sensitivity[1:d, 1:2d+2]

    return ϕ, pa, pb, N, ∂pa, ∂pb
end




# ################################################################################
# # demo
# ################################################################################
# vis = Visualizer()
# render(vis)
# set_floor!(vis)
# set_light!(vis)
# set_background!(vis)
#
# Aa = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ] .- 0.10ones(4,2)
# ba = 0.5*[
#     +1,
#     +1,
#     +1,
#      2,
#     ]
#
# Ab = [
#      1.0  0.0;
#      0.0  1.0;
#     -1.0  0.0;
#      0.0 -1.0;
#     ] .+ 0.10ones(4,2)
# bb = 0.5*[
#      1,
#      1,
#      1,
#      1,
#     ]
# na = length(ba)
# nb = length(bb)
#
# build_2d_polyhedron!(vis, Aa, ba, color=RGBA(0.2,0.2,0.2,0.6), name=:polya)
# build_2d_polyhedron!(vis, Ab, bb, color=RGBA(0.8,0.8,0.8,0.6), name=:polyb)
#
# xa2 = [0.4,3.0]
# xb2 = [0,4.0]
# qa2 = [+0.5]
# qb2 = [-0.5]
#
# set_2d_polyhedron!(vis, xa2, qa2, name=:polya)
# set_2d_polyhedron!(vis, xb2, qb2, name=:polyb)
#
# contact_solver = lp_contact_solver(Aa, ba, Ab, bb; d=2)
# set_pose_parameters!(contact_solver, xa2, qa2, xb2, qb2, na=na, nb=nb, d=d)
#
# solve!(contact_solver)
# p = contact_solver.solution.primals[1:d]
# ϕ = contact_solver.solution.primals[d+1]
#
# setobject!(vis[:contacta],
#     HyperSphere(GeometryBasics.Point(0, p...), 0.05),
#     MeshPhongMaterial(color=RGBA(1,0,0,1.0)))
#
# contact_bundle(contact_solver, xa2, qa2, xb2, qb2; na=na, nb=nb, d=d, differentiate=true)
