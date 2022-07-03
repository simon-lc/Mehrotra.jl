################################################################################
# residual
################################################################################
function contact_residual(primals, duals, slacks, parameters)

    d = length(primals)
    np = length(parameters)
    na = length(duals)
    nb = Int((np - na*(d+1) - 1 - 2d - 2*4)/(d+1))
    @show nb
    @show nb
    @show nb
    @show nb
    x1, q1, x2, q2, A1, b1, A2, b2, δ = unpack_contact_parameters(parameters, na=na, nb=nb, d=d)

    # y1 is expressed in body1's frame
    y1, z, s = primals, duals, slacks
    # yw is expressed in world frame
    yw = x1 + vector_rotate(y1, q1)
    # y2 is expressed in body2's frame
    y2 = vector_rotate(yw - x2, inv(q2))

    res = [
        gradient_squared_signed_distance(y2, A2, b2, δ) + A1' * z;
        # gradient_signed_distance(y2, A2, b2, δ2) + A1' * z;
        s - (-A1 * y1 + b1);
        # s .* z;
    ]
    return res
end

################################################################################
# parameters
################################################################################
function unpack_contact_parameters(parameters; na=1, nb=1, d=3)
    off = 0
    x1 = parameters[off .+ (1:d)]; off += d
    q1 = Quaternion(parameters[off .+ (1:4)]...); off += 4
    x2 = parameters[off .+ (1:d)]; off += d
    q2 = Quaternion(parameters[off .+ (1:4)]...); off += 4

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
    return [x1; vec(q1); x2; vec(q2); vec(A1); b1; vec(A2); b2; δ]
end

function num_contact_parameters(na, nb, d)
    2d + 2*4 + (na + nb) * (d+1) + 1
end

import Base.vec
function vec(q::Quaternion{T}) where T
    Vector{T}([q.s, q.v1, q.v2, q.v3])
end

# params0 = rand(47)
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
