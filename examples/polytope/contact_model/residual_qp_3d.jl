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
    # y1w = x1 + vector_rotate(y1, q1)
    # y2w = x2 + vector_rotate(y2, q1)
    y1w = x1 + z_rotation(q1) * y1
    y2w = x2 + z_rotation(q2) * y2

    res = [
        (y1w - y2w) + z_rotation(q1) * (A1' * z1);
        (y2w - y1w) + z_rotation(q2) * (A2' * z2);
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
function unpack_contact_parameters(parameters; na=1, nb=1, d=3)
    off = 0
    # x1 = parameters[off .+ (1:d)]; off += d
    # q1 = Quaternion(parameters[off .+ (1:4)]...); off += 4
    # x2 = parameters[off .+ (1:d)]; off += d
    # q2 = Quaternion(parameters[off .+ (1:4)]...); off += 4
    x1 = parameters[off .+ (1:d)]; off += d
    q1 = parameters[off .+ (1:3)]; off += 3 # MRP
    x2 = parameters[off .+ (1:d)]; off += d
    q2 = parameters[off .+ (1:3)]; off += 3 # MRP

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
    # return [x1; vec(q1); x2; vec(q2); vec(A1); b1; vec(A2); b2; δ]
    return [x1; q1; x2; q2; vec(A1); b1; vec(A2); b2; δ]
end

import Base.vec
function vec(q::Quaternion{T}) where T
    Vector{T}([q.s, q.v1, q.v2, q.v3])
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



# z_rotation([0,0,0.05]) * [1, 0, 0]

# params0 = rand(45)
# params1 = pack_contact_parameters(unpack_contact_parameters(params0, na=4, nb=4, d=3)...)
# norm(params0 - params1)

#
# y1 = rand(3)
# x1 = rand(3)
# q1 = Quaternion(normalize(rand(4))...)
#
# # yw is expressed in world frame
# yw = x1 + vector_rotate(y1, q1)
# # y2 is expressed in body2's frame
# y2 = vector_rotate(yw - x1, inv(q1))
#
