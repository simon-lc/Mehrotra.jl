################################################################################
# residual
################################################################################
function contact_residual(primals, duals, slacks, parameters; na::Int=0, nb::Int=0, d::Int=0)

    x1, q1, x2, q2, A1, b1, A2, b2, δ = unpack_contact_parameters(parameters, na=na, nb=nb, d=d)

    y, z, s = primals, duals, slacks
    z1 = z[1:na]
    z2 = z[na .+ (1:nb)]
    s1 = s[1:na]
    s2 = s[na .+ (1:nb)]

    # y1 is expressed in body1's frame
    y1 = y[1:d]
    # y2 is expressed in body2's frame
    y2 = y[d .+ (1:d)]
    # yw is expressed in world frame
    y1w = x1 + x_2d_rotation(q1) * y1
    y2w = x2 + x_2d_rotation(q2) * y2

    res = [
        (y1w - y2w) + x_2d_rotation(q1) * (A1' * z1);
        (y2w - y1w) + x_2d_rotation(q2) * (A2' * z2);
        s1 - (- A1 * y1 + b1);
        s2 - (- A2 * y2 + b2);
        # s1 .* z1;
        # s2 .* z2;
    ]
    return res
end

################################################################################
# parameters
################################################################################
function unpack_contact_parameters(parameters; na=0, nb=0, d=0)
    off = 0
    x1 = parameters[off .+ (1:d)]; off += d
    q1 = parameters[off .+ (1:3)]; off += 1
    x2 = parameters[off .+ (1:d)]; off += d
    q2 = parameters[off .+ (1:3)]; off += 1

    A1 = parameters[off .+ (1:na*d)]; off += na*d
    A1 = reshape(A1, (na,d))
    b1 = parameters[off .+ (1:na)]; off += na

    A2 = parameters[off .+ (1:nb*d)]; off += nb*d
    A2 = reshape(A2, (nb,d))
    b2 = parameters[off .+ (1:nb)]; off += nb
    δ = parameters[off + 1]; off += 1

    return x1, q1, x2, q2, A1, b1, A2, b2, δ
end

function pack_contact_parameters(x1, q1, x2, q2, A1, b1, A2, b2, δ)
    return [x1; q1; x2; q2; vec(A1); b1; vec(A2); b2; δ]
end

function x_2d_rotation(q)
    c = cos(q[1])
    s = sin(q[1])
    R = [c -s;
         s  c]
    return R
end

function z_rotation(q) # mrp
    θ = q[3]
    c = cos(θ)
    s = sin(θ)
    R = [c -s  0;
         s  c  0;
         0  0  1]
    return R
end
