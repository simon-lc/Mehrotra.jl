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

    # # pa is expressed in bodya's frame
    # pa = y[1:d]
    # l = y[d .+ (1:1)]
    # # pw is expressed in world's frame
    # pw = xa + x_2d_rotation(qa) * pa
    # # pb is expressed in bodyb's frame
    # pb = x_2d_rotation(qb)' * (pw - xb)

    # pw is expressed in world's frame
    pw = y[1:d]
    l = y[d .+ (1:1)]
    # pa is expressed in bodya's frame
    pa = x_2d_rotation(qa)' * (pw - xa)
    # pb is expressed in bodyb's frame
    pb = x_2d_rotation(qb)' * (pw - xb)

    res = [
        x_2d_rotation(qa) * Aa' * za + x_2d_rotation(qb) * Ab' * zb;
        1 - sum(za) - sum(zb);
        sa - (- Aa * pa + ba + l .* ones(na));
        sb - (- Ab * pb + bb + l .* ones(nb));
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
