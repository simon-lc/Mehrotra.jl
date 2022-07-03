################################################################################
# residual
################################################################################
function polytope_residual(primals, duals, slacks, parameters; na::Int=0, nb::Int=0, d::Int=0)

    xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb =
        unpack_polytope_parameters(parameters, na=na, nb=nb, d=d)

    y, z, s = primals, duals, slacks

    # duals
    za = z[1:na]
    zb = z[na .+ (1:nb)]
    γ = z[na + nb .+ (1:1)]

    # slacks
    sa = s[1:na]
    sb = s[na .+ (1:nb)]
    sγ = s[na + nb .+ (1:1)]

    # primals
    # pa is expressed in bodya's frame
    pa = y[1:d]
    # yb is expressed in bodyb's frame
    pb = y[d .+ (1:d)]

    # velocity
    va25 = y[2d .+ (1:d)]
    ωa25 = y[3d .+ (1:1)]
    vb25 = y[3d+1 .+ (1:d)]
    ωb25 = y[4d+1 .+ (1:1)]
    xa1 = xa2 - timestep .* va15
    qa1 = qa2 - timestep .* ωa15
    xb1 = xb2 - timestep .* vb15
    qb1 = qb2 - timestep .* ωb15
    xa3 = xa2 + timestep .* va25
    qa3 = qa2 + timestep .* ωa25
    xb3 = xb2 + timestep .* vb25
    qb3 = qb2 + timestep .* ωb25

    # pw is expressed in world frame
    paw = xa3 + x_2d_rotation(qa3) * pa
    pbw = xb3 + x_2d_rotation(qb3) * pb

    # controls
    ua = u[1:d+1]
    ub = u[d+1 .+ (1:d+1)]

    # signed distance function
    ϕ = [0.5 * (paw - pbw)'*(paw - pbw)]
    # Impact jacobien
    N = impact_jacobian(pa, pb, xa3, qa3, xb3, qb3, timestep)
    # mass matrix
    Ma = Diagonal([mass; mass; inertia])
    Mb = Diagonal([mass; mass; inertia])

    res = [
        (paw - pbw) + x_2d_rotation(qa3) * (Aa' * za);
        (pbw - paw) + x_2d_rotation(qb3) * (Ab' * zb);
        Ma * ([xa3; qa3] - 2*[xa2; qa2] + [xa1; qa1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ua * timestep[1] #- Na'*γ;
        Mb * ([xb3; qb3] - 2*[xb2; qb2] + [xb1; qb1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ub * timestep[1] #- Nb'*γ;
        sa - (- Aa * pa + ba);
        sb - (- Ab * pb + bb);
        sγ - ϕ;
        # sa .* za;
        # sb .* zb;
    ]
    @show size(res)
    @show size(N)
    res[1:2d+2d+2] += -N'*γ
    return res
end

function impact_jacobian(pa, pb, xa3, qa3, xb3, qb3, timestep)
    # ∂ϕ∂y
    # pw is expressed in world frame
    paw = xa3 + x_2d_rotation(qa3) * pa
    pbw = xb3 + x_2d_rotation(qb3) * pb

    ∂pa = (paw - pbw)' * x_2d_rotation(qa3)
    ∂pb = (pbw - paw)' * x_2d_rotation(qb3)
    ∂xa3 = (paw - pbw)'
    ∂va25 = timestep .* ∂xa3
    ∂xb3 = (pbw - paw)'
    ∂vb25 = timestep .* ∂xb3
    ∂qa3 = (paw - pbw)' * x_2d_rotation_jacobian(qa3) * pa
    ∂ωa25 = timeste################################################################################
# residual
################################################################################
function polytope_residual(primals, duals, slacks, parameters; na::Int=0, nb::Int=0, d::Int=0)

    xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb =
        unpack_polytope_parameters(parameters, na=na, nb=nb, d=d)

    y, z, s = primals, duals, slacks

    # duals
    za = z[1:na]
    zb = z[na .+ (1:nb)]
    γ = z[na + nb .+ (1:1)]

    # slacks
    sa = s[1:na]
    sb = s[na .+ (1:nb)]
    sγ = s[na + nb .+ (1:1)]

    # primals
    # pa is expressed in bodya's frame
    pa = y[1:d]
    # yb is expressed in bodyb's frame
    pb = y[d .+ (1:d)]

    # velocity
    va25 = y[2d .+ (1:d)]
    ωa25 = y[3d .+ (1:1)]
    vb25 = y[3d+1 .+ (1:d)]
    ωb25 = y[4d+1 .+ (1:1)]
    xa1 = xa2 - timestep .* va15
    qa1 = qa2 - timestep .* ωa15
    xb1 = xb2 - timestep .* vb15
    qb1 = qb2 - timestep .* ωb15
    xa3 = xa2 + timestep .* va25
    qa3 = qa2 + timestep .* ωa25
    xb3 = xb2 + timestep .* vb25
    qb3 = qb2 + timestep .* ωb25

    # pw is expressed in world frame
    paw = xa3 + x_2d_rotation(qa3) * pa
    pbw = xb3 + x_2d_rotation(qb3) * pb

    # controls
    ua = u[1:d+1]
    ub = u[d+1 .+ (1:d+1)]

    # signed distance function
    ϕ = [0.5 * (paw - pbw)'*(paw - pbw)]
    # Impact jacobien
    N = impact_jacobian(pa, pb, xa3, qa3, xb3, qb3, timestep)
    # mass matrix
    Ma = Diagonal([mass; mass; inertia])
    Mb = Diagonal([mass; mass; inertia])

    res = [
        (paw - pbw) + x_2d_rotation(qa3) * (Aa' * za);
        (pbw - paw) + x_2d_rotation(qb3) * (Ab' * zb);
        Ma * ([xa3; qa3] - 2*[xa2; qa2] + [xa1; qa1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ua * timestep[1] #- Na'*γ;
        Mb * ([xb3; qb3] - 2*[xb2; qb2] + [xb1; qb1])/timestep[1] - timestep[1] * [0; mass .* gravity; 0] - ub * timestep[1] #- Nb'*γ;
        sa - (- Aa * pa + ba);
        sb - (- Ab * pb + bb);
        sγ - ϕ;
        # sa .* za;
        # sb .* zb;
    ]
    @show size(res)
    @show size(N)
    res[1:2d+2d+2] += -N'*γ
    return res
end

function impact_jacobian(pa, pb, xa3, qa3, xb3, qb3, timestep)
    # ∂ϕ∂y
    # pw is expressed in world frame
    paw = xa3 + x_2d_rotation(qa3) * pa
    pbw = xb3 + x_2d_rotation(qb3) * pb

    ∂pa = (paw - pbw)' * x_2d_rotation(qa3)
    ∂pb = (pbw - paw)' * x_2d_rotation(qb3)
    ∂xa3 = (paw - pbw)'
    ∂va25 = timestep .* ∂xa3
    ∂xb3 = (pbw - paw)'
    ∂vb25 = timestep .* ∂xb3
    ∂qa3 = (paw - pbw)' * x_2d_rotation_jacobian(qa3) * pa
    ∂ωa25 = timestep .* ∂qa3
    ∂qb3 = (pbw - paw)' * x_2d_rotation_jacobian(qb3) * pb
    ∂ωb25 = timestep .* ∂qb3
    N = [∂pa ∂pb ∂va25 ∂ωa25 ∂vb25 ∂ωb25]
    return N
end


################################################################################
# parameters
################################################################################
function unpack_polytope_parameters(parameters; na=0, nb=0, d=0)
    off = 0
    xa = parameters[off .+ (1:d)]; off += d
    qa = parameters[off .+ (1:1)]; off += 1
    xb = parameters[off .+ (1:d)]; off += d
    qb = parameters[off .+ (1:1)]; off += 1

    va15 = parameters[off .+ (1:d)]; off += d
    ωa15 = parameters[off .+ (1:1)]; off += 1
    vb15 = parameters[off .+ (1:d)]; off += d
    ωb15 = parameters[off .+ (1:1)]; off += 1

    u = parameters[off .+ (1:2d+2)]; off += 2d+2
    timestep = parameters[off .+ (1:1)]; off += 1
    mass = parameters[off .+ (1:1)]; off += 1
    inertia = parameters[off .+ (1:1)]; off += 1
    gravity = parameters[off .+ (1:1)]; off += 1

    Aa = parameters[off .+ (1:na*d)]; off += na*d
    Aa = reshape(Aa, (na,d))
    ba = parameters[off .+ (1:na)]; off += na

    Ab = parameters[off .+ (1:nb*d)]; off += nb*d
    Ab = reshape(Ab, (nb,d))
    bb = parameters[off .+ (1:nb)]; off += nb

    return xa, qa, xb, qb, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb
end

function pack_polytope_parameters(xa, qa, xb, qb, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb)
    return [xa; qa; xb; qb; va15; ωa15; vb15; ωb15; u; timestep; mass; inertia; gravity; vec(Aa); ba; vec(Ab); bb]
end

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

function x_2d_rotation_jacobian(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [-s -c;
          c -s]
    return R
end

function simulate_2d_polytope(solver, xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U;
        timestep=0.01,
        mass=1.0,
        inertia=0.1,
        friction_coefficient=0.2,
        gravity=-9.81)

    H = length(U)
    Paw = []
    Pbw = []
    Xa = []
    Qa = []
    Xb = []
    Qb = []
    Va = []
    Vb = []
    Ωa = []
    Ωb = []
    iterations = Vector{Int}()

    paw = xa2 # fake
    pbw = xb2 # fake

    for i = 1:H
        push!(Paw, paw)
        push!(Pbw, pbw)
        push!(Xa, xa2)
        push!(Qa, qa2)
        push!(Xb, xb2)
        push!(Qb, qb2)
        push!(Va, va15)
        push!(Ωa, ωa15)
        push!(Vb, vb15)
        push!(Ωb, ωb15)
        parameters = pack_polytope_parameters(xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U[i], timestep, mass, inertia, gravity, Aa, ba, Ab, bb)
        solver.parameters .= parameters

        solver.options.verbose = false
        solve!(solver)
        push!(iterations, solver.trace.iterations)

        va15 .= solver.solution.primals[2d .+ (1:2)]
        ωa15 .= solver.solution.primals[3d .+ (1:1)]
        vb15 .= solver.solution.primals[3d+1 .+ (1:2)]
        ωb15 .= solver.solution.primals[4d+1 .+ (1:1)]
        xa2 = xa2 + timestep * va15
        qa2 = qa2 + timestep * ωa15
        xb2 = xb2 + timestep * vb15
        qb2 = qb2 + timestep * ωb15

        pa = solver.solution.primals[1:d]
        pb = solver.solution.primals[d .+ (1:d)]
        paw = xa2 + x_2d_rotation(qa2) * pa
        pbw = xb2 + x_2d_rotation(qb2) * pb
    end
    return Paw, Pbw, Xa, Qa, Xb, Qb, Va, Ωa, Vb, Ωb, iterations
end


#
#
# na = 4
# nb = 4
# d = 2
# θ0 = rand(46)
# θ1 = pack_polytope_parameters(
#     unpack_polytope_parameters(θ0, na=na, nb=nb, d=d)...
#     )
#
# norm(θ0 - θ1)
p .* ∂qa3
    ∂qb3 = (pbw - paw)' * x_2d_rotation_jacobian(qb3) * pb
    ∂ωb25 = timestep .* ∂qb3
    N = [∂pa ∂pb ∂va25 ∂ωa25 ∂vb25 ∂ωb25]
    return N
end


################################################################################
# parameters
################################################################################
function unpack_polytope_parameters(parameters; na=0, nb=0, d=0)
    off = 0
    xa = parameters[off .+ (1:d)]; off += d
    qa = parameters[off .+ (1:1)]; off += 1
    xb = parameters[off .+ (1:d)]; off += d
    qb = parameters[off .+ (1:1)]; off += 1

    va15 = parameters[off .+ (1:d)]; off += d
    ωa15 = parameters[off .+ (1:1)]; off += 1
    vb15 = parameters[off .+ (1:d)]; off += d
    ωb15 = parameters[off .+ (1:1)]; off += 1

    u = parameters[off .+ (1:2d+2)]; off += 2d+2
    timestep = parameters[off .+ (1:1)]; off += 1
    mass = parameters[off .+ (1:1)]; off += 1
    inertia = parameters[off .+ (1:1)]; off += 1
    gravity = parameters[off .+ (1:1)]; off += 1

    Aa = parameters[off .+ (1:na*d)]; off += na*d
    Aa = reshape(Aa, (na,d))
    ba = parameters[off .+ (1:na)]; off += na

    Ab = parameters[off .+ (1:nb*d)]; off += nb*d
    Ab = reshape(Ab, (nb,d))
    bb = parameters[off .+ (1:nb)]; off += nb

    return xa, qa, xb, qb, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb
end

function pack_polytope_parameters(xa, qa, xb, qb, va15, ωa15, vb15, ωb15, u, timestep, mass, inertia, gravity, Aa, ba, Ab, bb)
    return [xa; qa; xb; qb; va15; ωa15; vb15; ωb15; u; timestep; mass; inertia; gravity; vec(Aa); ba; vec(Ab); bb]
end

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

function x_2d_rotation_jacobian(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [-s -c;
          c -s]
    return R
end

function simulate_2d_polytope(solver, xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U;
        timestep=0.01,
        mass=1.0,
        inertia=0.1,
        friction_coefficient=0.2,
        gravity=-9.81)

    H = length(U)
    Paw = []
    Pbw = []
    Xa = []
    Qa = []
    Xb = []
    Qb = []
    Va = []
    Vb = []
    Ωa = []
    Ωb = []
    iterations = Vector{Int}()

    paw = xa2 # fake
    pbw = xb2 # fake

    for i = 1:H
        push!(Paw, paw)
        push!(Pbw, pbw)
        push!(Xa, xa2)
        push!(Qa, qa2)
        push!(Xb, xb2)
        push!(Qb, qb2)
        push!(Va, va15)
        push!(Ωa, ωa15)
        push!(Vb, vb15)
        push!(Ωb, ωb15)
        parameters = pack_polytope_parameters(xa2, qa2, xb2, qb2, va15, ωa15, vb15, ωb15, U[i], timestep, mass, inertia, gravity, Aa, ba, Ab, bb)
        solver.parameters .= parameters

        solver.options.verbose = false
        solve!(solver)
        push!(iterations, solver.trace.iterations)

        va15 .= solver.solution.primals[2d .+ (1:2)]
        ωa15 .= solver.solution.primals[3d .+ (1:1)]
        vb15 .= solver.solution.primals[3d+1 .+ (1:2)]
        ωb15 .= solver.solution.primals[4d+1 .+ (1:1)]
        xa2 = xa2 + timestep * va15
        qa2 = qa2 + timestep * ωa15
        xb2 = xb2 + timestep * vb15
        qb2 = qb2 + timestep * ωb15

        pa = solver.solution.primals[1:d]
        pb = solver.solution.primals[d .+ (1:d)]
        paw = xa2 + x_2d_rotation(qa2) * pa
        pbw = xb2 + x_2d_rotation(qb2) * pb
    end
    return Paw, Pbw, Xa, Qa, Xb, Qb, Va, Ωa, Vb, Ωb, iterations
end


#
#
# na = 4
# nb = 4
# d = 2
# θ0 = rand(46)
# θ1 = pack_polytope_parameters(
#     unpack_polytope_parameters(θ0, na=na, nb=nb, d=d)...
#     )
#
# norm(θ0 - θ1)
