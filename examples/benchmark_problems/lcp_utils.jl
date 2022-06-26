function unpack_lcp_non_negative_cone_parameters(parameters, num_primals, num_cone)
    off = 0
    A = reshape(parameters[off .+ (1:num_primals^2)], num_primals, num_primals)
    off += num_primals^2
    b = reshape(parameters[off .+ (1:num_primals)], num_primals)
    off += num_primals
    C = reshape(parameters[off .+ (1:num_cone^2)], num_cone, num_cone)
    off += num_cone^2
    d = reshape(parameters[off .+ (1:num_cone)], num_cone)
    off += num_cone
    return A, b, C, d
end

function lcp_non_negative_cone_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_primals = length(primals)
    num_cone = length(duals)
    A, b, C, d = unpack_lcp_non_negative_cone_parameters(parameters, num_primals, num_cone)

    res = [
        A * y + b;
        C * z + d - s;
        # z .* s .- κ[1];
        ]
    return res
end

function unpack_lcp_second_order_cone_parameters(parameters, num_primals, num_cone)
    off = 0
    A = reshape(parameters[off .+ (1:num_primals^2)], num_primals, num_primals)
    off += num_primals^2
    b = reshape(parameters[off .+ (1:num_primals)], num_primals)
    off += num_primals
    C = reshape(parameters[off .+ (1:num_cone^2)], num_cone, num_cone)
    off += num_cone^2
    d = reshape(parameters[off .+ (1:num_cone)], num_cone)
    off += num_cone
    return A, b, C, d
end

function lcp_second_order_cone_residual(primals, duals, slacks, parameters)
    y, z, s = primals, duals, slacks
    num_primals = length(primals)
    num_cone = length(duals)
    A, b, C, d = unpack_lcp_second_order_cone_parameters(parameters, num_primals, num_cone)

    res = [
        A * y + b;
        C * z + d - s;
        # z .* s .- κ[1];
        ]
    return res
end
