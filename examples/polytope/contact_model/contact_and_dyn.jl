################################################################################
# residual
################################################################################
function joint_residual(primals, duals, slacks, parameters; np::Int=0, nc::Int=0, d::Int=0)
    xp2, xc2, vp15, vc15, Ap, bp, Ac, bc = unpack_joint_parameters(parameters, np=np, nc=nc, d=d)
    up = zeros(3)
    uc = zeros(3)
    timestep = 0.01
    mass = 1.0
    inertia = 0.2
    gravity = -9.81
    friction_coefficient = 0.2

    y, z, s = primals, duals, slacks
    γ = z[1:1]
    zp = z[1 .+ (1:np)]
    zc = z[1+np .+ (1:nc)]
    sγ = s[1:1]
    sp = s[1 .+ (1:np)]
    sc = s[1+np .+ (1:nc)]

    # pw is expressed in world's frame
    off = 0
    vp25 = y[off .+ (1:d+1)]; off += d+1
    vc25 = y[off .+ (1:d+1)]; off += d+1
    xp3 = xp2 + timestep * vp25
    xc3 = xc2 + timestep * vc25
    xp1 = xp2 - timestep * vp15
    xc1 = xc2 - timestep * vc15
    vc25 = y[off .+ (1:d+1)]; off += d+1
    pw = y[off .+ (1:d)] + (xp3[1:2] + xc3[1:2]) ./ 2; off += d
    ϕ = y[off .+ (1:1)]; off += 1

    # pp is expressed in pbody's frame
    pp = x_2d_rotation(xp3[3:3])' * (pw - xp3[1:2])
    # pc is expressed in cbody's frame
    pc = x_2d_rotation(xc3[3:3])' * (pw - xc3[1:2])

    Np =
    Nc =

    res = [
        mass * (xp3 - 2xp2 + xp1)/timestep - timestep * mass * [0,0, gravity] - Np' * γ;
        mass * (xc3 - 2xc2 + xc1)/timestep - timestep * mass * [0,0, gravity] - Nc' * γ;
        x_2d_rotation(xp3[3:3]) * Ap' * zp + x_2d_rotation(xc3[3:3]) * Ac' * zc;
        1 - sum(zp) - sum(zc);
        sγ - ϕ;
        sp - (- Ap * pp + bp + ϕ .* ones(np));
        sc - (- Ac * pc + bc + ϕ .* ones(nc));
        # sp .* zp;
        # sc .* zc;
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
function unpack_joint_parameters(parameters; np=0, nc=0, d=0)
    off = 0
    xp = parameters[off .+ (1:d+1)]; off += d+1
    xc = parameters[off .+ (1:d+1)]; off += d+1
    vp = parameters[off .+ (1:d+1)]; off += d+1
    vc = parameters[off .+ (1:d+1)]; off += d+1

    Ap = parameters[off .+ (1:np*d)]; off += np*d
    Ap = reshape(Ap, (np,d))
    bp = parameters[off .+ (1:np)]; off += np

    Ac = parameters[off .+ (1:nc*d)]; off += nc*d
    Ac = reshape(Ac, (nc,d))
    bc = parameters[off .+ (1:nc)]; off += nc

    return xp, xc, vp, vc, Ap, bp, Ac, bc
end

function pack_joint_parameters(xp, xc, vp, vc, Ap, bp, Ac, bc)
    return [xp; xc; vp; vc; vec(Ap); bp; vec(Ac); bc]
end


################################################################################
# demo
################################################################################
# parameters
Ap = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .- 0.00ones(4,2)
bp = 0.5*[
    +1,
    +1,
    +1,
     1,
     # 2,
    ]
Ac = [
     1.0  0.0;
     0.0  1.0;
    -1.0  0.0;
     0.0 -1.0;
    ] .+ 0.00ones(4,2)
bc = 0.5*[
     1,
     1,
     1,
     1,
    ]

timestep = 0.05
gravity = -9.81
mass = 1.0
inertia = 0.2 * ones(1,1)

np = length(bp)
nc = length(bc)
d = 2

xp2 = zeros(d+1)
xc2 = zeros(d+1)
vp15 = zeros(d+1)
vc15 = zeros(d+1)

parameters = pack_joint_parameters(xp2, xc2, vp15, vc15, Ap, bp, Ac, bc)
num_primals = 3*(d + 1)
num_cone = np + nc
idx_nn = collect(1:num_cone)
idx_soc = [collect(1:0)]

sized_residual(primals, duals, slacks, parameters) =
    joint_residual(primals, duals, slacks, parameters; np=np, nc=nc, d=d)

solver = Solver(
        sized_residual,
        num_primals,
        num_cone,
        parameters=parameters,
        nonnegative_indices=idx_nn,
        second_order_indices=idx_soc,
        options=Options(complementarity_tolerance=3e-3),
        )
